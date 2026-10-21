"""FCRA-G: delayed-label anchored online graph correction on PACS."""

from collections import Counter, OrderedDict
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.delayed_feedback.simatta import write_csv
from experiments.fcra.algorithm import DEFAULTS as FCRA_DEFAULTS, FCRA


DEFAULTS = {
    **FCRA_DEFAULTS,
    "graph_memory_size": 256,
    "graph_anchor_limit": 128,
    "graph_neighbors": 8,
    "graph_tau": 0.1,
    "graph_lambda": 1.0,
    "graph_epsilon": 1e-12,
    "expert_prior": [0.6, 0.3, 0.1],
    "expert_eta": 0.1,
    "evidence_decay": 0.99,
    "expert1_beta": [0.10, 0.40],
    "expert2_beta": [0.25, 0.75],
    "score_epsilon": 1e-12,
}


GRAPH_HEADER = (
    "step,domain,samples,main_accuracy_pct,final_accuracy_pct,graph1_accuracy_pct,"
    "graph2_accuracy_pct,main_graph1_disagreement_pct,main_graph2_disagreement_pct,"
    "main_final_disagreement_pct,graph1_graph2_disagreement_pct,changed,changed_correct,"
    "changed_incorrect,h_mean,h_zero_pct,w0,w1,w2,beta1_mean,beta2_mean,nodes,anchors,"
    "unlabeled,memory_nodes,edges,graph_seconds,solve_seconds,fallback"
)

EXPERT_HEADER = (
    "step,anchor_index,sample_id,query_step,return_step,delay,query_domain,"
    "return_domain,label,p0_true,p1_true,p2_true,output_true,loss0,loss1,loss2,"
    "gain0,gain1,gain2,w0,w1,w2"
)


class GraphFCRA(FCRA):
    """Keep FCRA's supervised path and add a non-parametric graph prediction path."""

    def __init__(self, config, delay, trace=None):
        super().__init__(config, delay)
        self.trace = trace
        self.reference_mode = trace is not None
        if self.reference_mode:
            structure = [
                (self.experiment.domains[env],
                 min(config.atta.batch_size, len(loader.sampler) - start))
                for env in config.dataset.test_envs[1:]
                for loader in [self.fast_loader[env]]
                for start in range(0, len(loader.sampler), config.atta.batch_size)
            ]
            expected = [(row["domain"], int(row["samples"])) for row in trace.batches]
            if structure != expected:
                raise ValueError("Current dataset stream and reference batch structure differ")
        self.graph_enabled = delay > 0
        self.unlabeled_memory = OrderedDict()
        self.query_expert_predictions = {}
        self.scored_queries = set()
        self.expert_prior = torch.tensor(DEFAULTS["expert_prior"], dtype=torch.float32)
        self.expert_evidence = torch.zeros(3, dtype=torch.float32)
        self.current_expert_weights = self.expert_prior.clone()
        self.graph_rows, self.expert_event_rows = [], []
        self.graph_seconds = 0.0
        self.solve_seconds = 0.0
        self.training_seconds = 0.0
        self.graph_batches = 0
        self.graph_node_total = 0
        self.graph_edge_total = 0
        self.main_forward_calls = 0
        self.teacher_forward_calls = 0
        self.main_model_calls = 0
        self.teacher_model_calls = 0
        self.memory_forward_calls = 0
        self.memory_forward_images = 0
        self.current_expert_probs = None
        self.current_output_probs = None
        self.current_h = None
        self.current_graph_stats = None
        self.model.register_forward_hook(self._count_main_model_forward)
        self.teacher.register_forward_hook(self._count_teacher_model_forward)
        self.model[0].register_forward_hook(self._count_main_forward)
        self.teacher[0].register_forward_hook(self._count_teacher_forward)

    def _count_main_model_forward(self, module, inputs, output):
        self.main_model_calls += 1

    def _count_teacher_model_forward(self, module, inputs, output):
        self.teacher_model_calls += 1

    def _count_main_forward(self, module, inputs, output):
        self.main_forward_calls += 1

    def _count_teacher_forward(self, module, inputs, output):
        self.teacher_forward_calls += 1

    def _decay_and_compute_expert_weights(self):
        if not self.graph_enabled:
            self.current_expert_weights = torch.tensor([1.0, 0.0, 0.0])
            return
        self.expert_evidence.mul_(DEFAULTS["evidence_decay"])
        logits = torch.log(self.expert_prior) + DEFAULTS["expert_eta"] * self.expert_evidence
        self.current_expert_weights = torch.softmax(logits, dim=0)

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        if self.reference_mode:
            return self._select_queries_from_trace(data, stream_ids)
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        # The inherited FCRA path fixes the query plan and the calibration
        # teacher cache first; graph correction never changes query IDs/times.
        outputs = super().select_queries(data, stream_ids)
        if not self.graph_enabled:
            return outputs
        self._decay_and_compute_expert_weights()
        main_probs = outputs.softmax(1)
        self.teacher.eval()
        source_raw = self.teacher[0](data)
        source_features = F.normalize(source_raw, dim=1)
        corrected = self._graph_correct(data, stream_ids, main_probs, source_features)
        current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        if current > previous:
            self._save_query_predictions(previous, current, stream_ids)
        self._add_unlabeled_batch(data, stream_ids, source_features)
        return corrected

    @torch.no_grad()
    def _select_queries_from_trace(self, data, stream_ids):
        """Replay the recorded FCRA query IDs and timing exactly."""
        self.stream_step += 1
        self.current_data = data.detach()
        self.model.eval()
        features = self.model[0](data)
        outputs = self.model[1](features)
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        records = self.trace.queries[self.stream_step]
        expected = list(range(previous, previous + len(records)))
        if [int(row["anchor_index"]) for row in records] != expected:
            raise ValueError("Reference anchor insertion order changed")
        if any(int(row["query_step"]) != self.stream_step for row in records):
            raise ValueError("Reference query step differs from the stream clock")
        first_stream_id = int(stream_ids[0])
        positions = torch.tensor([int(row["sample_id"]) - first_stream_id
                                  for row in records], dtype=torch.long)
        if len(positions) and (int(positions.min()) < 0 or int(positions.max()) >= len(data)):
            raise ValueError("Reference query is outside the current batch")
        if records:
            device_positions = positions.to(data.device)
            self.query_anchors = self.update_anchors(
                self.query_anchors, data[device_positions].cpu(), stream_ids[positions],
                features[device_positions].cpu(), torch.ones(len(positions)))
            self.teacher.eval()
            teacher_raw = self.teacher[0](data[device_positions])
            teacher_features = F.normalize(teacher_raw, dim=1)
            teacher_predictions = self.teacher[1](teacher_raw).argmax(1)
            for offset, index in enumerate(expected):
                self.teacher_predictions[index] = int(teacher_predictions[offset])
                self.teacher_features[index] = teacher_features[offset].cpu()
        current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        pending = sum(1 for index in range(previous) if index not in self.received)
        self.query_rows.append({
            "step": self.stream_step, "pending_before_release": pending,
            "eligible_before": None, "eligible_after": None, "suppressed": None,
            "new_queries": current - previous, "mean_redundancy": None,
        })
        if not self.graph_enabled:
            return outputs
        self._decay_and_compute_expert_weights()
        main_probs = outputs.softmax(1)
        self.teacher.eval()
        source_raw = self.teacher[0](data)
        source_features = F.normalize(source_raw, dim=1)
        corrected = self._graph_correct(data, stream_ids, main_probs, source_features)
        if current > previous:
            self._save_query_predictions(previous, current, stream_ids)
        self._add_unlabeled_batch(data, stream_ids, source_features)
        return corrected

    @torch.no_grad()
    def _memory_probabilities(self, images):
        size = self.config.atta.batch_size
        probabilities = []
        for start in range(0, len(images), size):
            batch = images[start:start + size].to(self.config.device)
            probabilities.append(self.model(batch).softmax(1).cpu())
            self.memory_forward_calls += 1
            self.memory_forward_images += len(batch)
        return torch.cat(probabilities, dim=0)

    def _select_anchor_indices(self):
        indices = sorted(self.received)
        if not indices:
            return []
        classes = self.config.dataset.num_classes
        by_class = {label: [] for label in range(classes)}
        for index in indices:
            label = int(self.received[index])
            if label not in by_class:
                continue
            record = self.oracle.records[index]
            return_step = record.return_step if record.return_step is not None else -1
            by_class[label].append((return_step, index))
        for label in by_class:
            # Most recently returned first; anchor index breaks ties.
            by_class[label].sort(key=lambda item: (-item[0], -item[1]))
        limit = DEFAULTS["graph_anchor_limit"]
        base = limit // classes
        remainder = limit % classes
        quotas = [base + (1 if label < remainder else 0) for label in range(classes)]
        selected = []
        leftovers = {}
        for label in range(classes):
            items = by_class[label]
            take = min(quotas[label], len(items))
            selected.extend(index for _, index in items[:take])
            if len(items) > take:
                leftovers[label] = items[take:]
        while len(selected) < limit:
            advanced = False
            for label in range(classes):
                items = leftovers.get(label)
                if items:
                    _, index = items.pop(0)
                    selected.append(index)
                    advanced = True
                    if len(selected) >= limit:
                        break
            if not advanced:
                break
        return selected

    def _anchor_features(self, indices, feature_dim):
        features = []
        for index in indices:
            feature = self.teacher_features.get(index)
            if feature is None:
                image = self.query_anchors.data[index].unsqueeze(0).to(self.config.device)
                feature = F.normalize(self.teacher[0](image), dim=1)[0].cpu()
                self.teacher_features[index] = feature
            features.append(feature)
        if not features:
            return torch.empty((0, feature_dim), dtype=torch.float32)
        return torch.stack(features)

    def _solve_graph(self, anchor_features, anchor_labels, u_features, P_U):
        n_anchors = len(anchor_labels)
        n_unlabeled = len(u_features)
        classes = P_U.shape[1]
        if n_anchors:
            Y_L = F.one_hot(anchor_labels, num_classes=classes).float()
        else:
            Y_L = torch.empty((0, classes), dtype=torch.float32)
        node_features = torch.cat([anchor_features, u_features], dim=0)
        nodes = len(node_features)
        W = torch.zeros((nodes, nodes), dtype=torch.float32)
        if nodes > 1:
            similarity = node_features @ node_features.T
            similarity.fill_diagonal_(-1.0)
            positive_similarity = similarity.masked_fill(similarity <= 0, -1.0)
            neighbours = min(DEFAULTS["graph_neighbors"], nodes - 1)
            values, positions = positive_similarity.topk(neighbours, dim=1)
            positive = (values > 0).float()
            weights = torch.exp((values - 1.0) / DEFAULTS["graph_tau"]) * positive
            W.scatter_(1, positions, weights)
            W = torch.maximum(W, W.T)
        degrees = W.sum(1)
        W = W / max(1.0, float(degrees.max()))
        degrees = W.sum(1)
        fallback = False
        solve_seconds = 0.0
        Q_U = P_U.clone()
        h = torch.zeros(n_unlabeled, dtype=P_U.dtype)
        if n_unlabeled:
            W_UU = W[n_anchors:, n_anchors:]
            W_UL = W[n_anchors:, :n_anchors]
            laplacian = torch.diag(degrees[n_anchors:]) - W_UU
            system = torch.eye(n_unlabeled) + DEFAULTS["graph_lambda"] * laplacian
            rhs = torch.cat([
                P_U + DEFAULTS["graph_lambda"] * (W_UL @ Y_L),
                DEFAULTS["graph_lambda"] * W_UL.sum(1, keepdim=True),
            ], dim=1)
            try:
                solve_start = time.perf_counter()
                solution = torch.linalg.solve(system.double(), rhs.double())
                solve_seconds = time.perf_counter() - solve_start
                Q_U = solution[:, :classes].float()
                h = solution[:, classes].float()
                if not bool(torch.isfinite(Q_U).all() and torch.isfinite(h).all()):
                    fallback = True
                else:
                    Q_U = Q_U.clamp_min(0)
                    row_sums = Q_U.sum(1, keepdim=True)
                    valid = row_sums > DEFAULTS["graph_epsilon"]
                    Q_U = torch.where(
                        valid,
                        Q_U / row_sums.clamp_min(DEFAULTS["graph_epsilon"]),
                        P_U,
                    )
                    h = h.clamp(0, 1)
            except RuntimeError:
                fallback = True
        if fallback:
            Q_U = P_U.clone()
            h = torch.zeros(n_unlabeled, dtype=P_U.dtype)
        return Q_U, h, {
            "nodes": int(nodes), "anchors": int(n_anchors),
            "unlabeled": int(n_unlabeled), "edges": int((W > 0).sum()) // 2,
            "solve_seconds": solve_seconds, "fallback": fallback,
        }

    @torch.no_grad()
    def _graph_correct(self, data, stream_ids, main_probs, source_features):
        started = time.perf_counter()
        # Graph construction and memory-prior refresh must not change BN/dropout.
        self.model.eval()
        self.teacher.eval()
        current_features = source_features.detach().cpu()
        current_probs = main_probs.detach().cpu()
        current_ids = stream_ids.tolist()
        current_set = set(current_ids)
        memory_items = [(sample_id, image, feature)
                        for sample_id, (image, feature) in self.unlabeled_memory.items()
                        if sample_id not in current_set]
        if memory_items:
            memory_images = torch.stack([item[1] for item in memory_items])
            memory_features = torch.stack([item[2] for item in memory_items])
            memory_probs = self._memory_probabilities(memory_images)
        else:
            memory_features = torch.empty((0, current_features.shape[1]),
                                          dtype=current_features.dtype)
            memory_probs = torch.empty((0, current_probs.shape[1]),
                                       dtype=current_probs.dtype)
        u_features = torch.cat([current_features, memory_features], dim=0)
        P_U = torch.cat([current_probs, memory_probs], dim=0)
        anchor_indices = self._select_anchor_indices()
        anchor_labels = torch.tensor([self.received[index] for index in anchor_indices],
                                     dtype=torch.long)
        anchor_features = self._anchor_features(anchor_indices, current_features.shape[1])
        Q_U, h, stats = self._solve_graph(anchor_features, anchor_labels, u_features, P_U)
        batch = len(current_probs)
        P_cur, Q_cur, h_cur = P_U[:batch], Q_U[:batch], h[:batch]
        weights = self.current_expert_weights.to(P_cur.dtype)
        beta1_range = DEFAULTS["expert1_beta"]
        beta2_range = DEFAULTS["expert2_beta"]
        beta1 = (beta1_range[0] + beta1_range[1] * h_cur).clamp(0, 1)
        beta2 = (beta2_range[0] + beta2_range[1] * h_cur).clamp(0, 1)
        p1 = (1 - beta1[:, None]) * P_cur + beta1[:, None] * Q_cur
        p2 = (1 - beta2[:, None]) * P_cur + beta2[:, None] * Q_cur
        p_out = weights[0] * P_cur + weights[1] * p1 + weights[2] * p2
        p_out = p_out.clamp_min(0)
        p_out = p_out / p_out.sum(1, keepdim=True).clamp_min(DEFAULTS["graph_epsilon"])
        stats.update({
            "graph_seconds": time.perf_counter() - started,
            "h_mean": float(h_cur.mean()),
            "h_zero_pct": 100.0 * float((h_cur <= 1e-8).float().mean()),
            "memory_nodes": len(memory_items),
            "current_nodes": batch,
            "w0": float(weights[0]), "w1": float(weights[1]), "w2": float(weights[2]),
            "beta1_mean": float(beta1.mean()), "beta2_mean": float(beta2.mean()),
        })
        self.current_expert_probs = [P_cur, p1, p2]
        self.current_output_probs = p_out
        self.current_h = h_cur
        self.current_graph_stats = stats
        self.graph_seconds += stats["graph_seconds"]
        self.solve_seconds += stats["solve_seconds"]
        self.graph_batches += 1
        self.graph_node_total += stats["nodes"]
        self.graph_edge_total += stats["edges"]
        return p_out.to(main_probs.device)

    def _save_query_predictions(self, previous, current, stream_ids):
        if not self.graph_enabled or current <= previous:
            return
        first_stream_id = int(stream_ids[0])
        batch = len(self.current_output_probs)
        for index in range(previous, current):
            sample_id = int(self.query_anchors.target[index])
            local = sample_id - first_stream_id
            if not 0 <= local < batch:
                continue
            self.query_expert_predictions[index] = {
                "sample_id": sample_id,
                "query_step": self.stream_step,
                "experts": tuple(prob[local].clone() for prob in self.current_expert_probs),
                "output": self.current_output_probs[local].clone(),
                "weights": self.current_expert_weights.clone(),
            }

    def _add_unlabeled_batch(self, data, stream_ids, source_features):
        if not self.graph_enabled:
            return
        for local, sample_id in enumerate(stream_ids.tolist()):
            self.unlabeled_memory[sample_id] = (
                data[local].detach().cpu(), source_features[local].detach().cpu())
            self.unlabeled_memory.move_to_end(sample_id)
        while len(self.unlabeled_memory) > DEFAULTS["graph_memory_size"]:
            self.unlabeled_memory.popitem(last=False)

    @torch.no_grad()
    def _score_expert_feedback(self):
        if not self.graph_enabled:
            return
        new_indices = sorted(set(self.received) - self.scored_queries)
        gains = torch.zeros(3, dtype=torch.float32)
        for index in new_indices:
            self.scored_queries.add(index)
            sample_id = int(self.query_anchors.target[index])
            self.unlabeled_memory.pop(sample_id, None)
            saved = self.query_expert_predictions.get(index)
            if saved is None:
                continue
            label = int(self.received[index])
            probabilities = [float(prob[label]) for prob in saved["experts"]]
            epsilon = DEFAULTS["score_epsilon"]
            losses = [-math.log(max(probability, epsilon)) for probability in probabilities]
            gain_values = [min(1.0, max(-1.0, losses[0] - loss)) for loss in losses]
            for expert in range(3):
                gains[expert] += gain_values[expert]
            record = self.oracle.records[index]
            weights = saved["weights"]
            self.expert_event_rows.append({
                "step": self.stream_step, "anchor_index": index, "sample_id": sample_id,
                "query_step": record.query_step, "return_step": record.return_step,
                "delay": (record.return_step - record.query_step
                          if record.return_step is not None else self.oracle.delay),
                "query_domain": record.query_domain, "return_domain": record.return_domain,
                "label": label,
                "p0_true": probabilities[0], "p1_true": probabilities[1],
                "p2_true": probabilities[2],
                "output_true": float(saved["output"][label]),
                "loss0": losses[0], "loss1": losses[1], "loss2": losses[2],
                "gain0": gain_values[0], "gain1": gain_values[1],
                "gain2": gain_values[2],
                "w0": float(weights[0]), "w1": float(weights[1]), "w2": float(weights[2]),
            })
        self.expert_evidence += gains

    @torch.enable_grad()
    def _train_supervised_quota(self, quota):
        """Run exactly the recorded number of FCRA supervised updates."""
        if quota <= 0:
            return
        indices, data, labels = self._supervised_memory()
        if not indices:
            return
        need, relevance, score, probabilities = self._replay_statistics(data, labels)
        reference = self.source_anchors
        has_pseudo = reference is not None and reference.num_elem() > 0
        pseudo_count = reference.num_elem() if has_pseudo else 0
        returned = len(indices)
        alpha = returned / (returned + pseudo_count) if has_pseudo else 1.0
        if has_pseudo and pseudo_count < self.cold_start:
            alpha = min(0.2, alpha)
        size = self.config.train.train_bs
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.config.atta.SimATTA.lr, momentum=0.9)
        draws = torch.zeros(returned)
        for iteration in range(quota):
            sampled = torch.multinomial(probabilities, size, replacement=True)
            images = data[sampled].to(self.config.device)
            targets = labels[sampled].to(self.config.device)
            draws.index_add_(0, sampled, torch.ones(size))
            losses = F.cross_entropy(self.model(images), targets, reduction="none")
            supervised_loss = losses.mean()
            pseudo_loss, gates = None, None
            if has_pseudo:
                pseudo_indices = torch.randint(0, pseudo_count, (size,))
                pseudo_labels = reference.target[pseudo_indices].to(
                    self.config.device).long()
                teacher_features = F.normalize(
                    reference.feats[pseudo_indices].to(self.config.device), dim=1)
                gates = self._feedback_gates(teacher_features, pseudo_labels)
                pseudo_loss = (gates * F.cross_entropy(
                    self.model(reference.data[pseudo_indices].to(self.config.device)),
                    pseudo_labels, reduction="none")).mean()
                loss = alpha * supervised_loss + (1 - alpha) * pseudo_loss
            else:
                loss = supervised_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.update_rows.append({
                "step": self.stream_step, "iteration": iteration,
                "returned_memory": returned, "replay_batch": int(len(sampled)),
                "source_memory": pseudo_count, "alpha": alpha,
                "has_pseudo": has_pseudo,
                "supervised_loss": float(supervised_loss.detach()),
                "pseudo_loss": float(pseudo_loss.detach()) if has_pseudo else None,
                "gate_mean": float(gates.mean()) if has_pseudo else None,
                "gate_min": float(gates.min()) if has_pseudo else None,
                "gate_max": float(gates.max()) if has_pseudo else None,
                "drawn_probability_mean": float(probabilities[sampled].mean()),
                "calibration_cache": len(self.calibration_events),
            })
        distinct = int((draws > 0).sum())
        self.replay_rows.append({
            "step": self.stream_step, "returned_memory": returned,
            "updates": quota, "source_memory": pseudo_count,
            "need_mean": float(need.mean()), "need_max": float(need.max()),
            "relevance_mean": float(relevance.mean()),
            "score_mean": float(score.mean()), "score_max": float(score.max()),
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "effective_sample_size": float(1 / probabilities.square().sum()),
            "distinct_sampled": distinct,
            "coverage": distinct / returned,
            "max_draws": int(draws.max()),
        })

    def cluster_train(self, target_anchors, source_anchors):
        self._score_expert_feedback()
        started = time.perf_counter()
        if self.reference_mode:
            self._record_calibration_events()
            self._train_supervised_quota(self.trace.old_updates[self.stream_step])
            result = None
        else:
            result = super().cluster_train(target_anchors, source_anchors)
        self.training_seconds += time.perf_counter() - started
        return result

    def on_batch_end(self, step, domain, data, target, outputs, released):
        if not self.graph_enabled or self.current_expert_probs is None:
            return
        target = target.cpu()
        p0, p1, p2 = [prob.detach().cpu() for prob in self.current_expert_probs]
        p_out = self.current_output_probs.detach().cpu()
        if len(target) != len(p0):
            raise ValueError("Graph diagnostics received a mismatched batch")
        pred0, pred1, pred2, pred_out = (p0.argmax(1), p1.argmax(1),
                                         p2.argmax(1), p_out.argmax(1))
        correct0, correct1, correct2, correct_out = (
            pred0 == target, pred1 == target, pred2 == target, pred_out == target)
        changed = pred_out != pred0
        stats = self.current_graph_stats
        self.graph_rows.append({
            "step": step, "domain": domain, "samples": len(target),
            "main_accuracy_pct": 100.0 * float(correct0.float().mean()),
            "final_accuracy_pct": 100.0 * float(correct_out.float().mean()),
            "graph1_accuracy_pct": 100.0 * float(correct1.float().mean()),
            "graph2_accuracy_pct": 100.0 * float(correct2.float().mean()),
            "main_graph1_disagreement_pct": 100.0 * float((pred0 != pred1).float().mean()),
            "main_graph2_disagreement_pct": 100.0 * float((pred0 != pred2).float().mean()),
            "main_final_disagreement_pct": 100.0 * float(changed.float().mean()),
            "graph1_graph2_disagreement_pct": 100.0 * float((pred1 != pred2).float().mean()),
            "changed": int(changed.sum()),
            "changed_correct": int((changed & correct_out).sum()),
            "changed_incorrect": int((changed & ~correct_out).sum()),
            "h_mean": stats["h_mean"], "h_zero_pct": stats["h_zero_pct"],
            "w0": stats["w0"], "w1": stats["w1"], "w2": stats["w2"],
            "beta1_mean": stats["beta1_mean"], "beta2_mean": stats["beta2_mean"],
            "nodes": stats["nodes"], "anchors": stats["anchors"],
            "unlabeled": stats["unlabeled"], "memory_nodes": stats["memory_nodes"],
            "edges": stats["edges"], "graph_seconds": stats["graph_seconds"],
            "solve_seconds": stats["solve_seconds"], "fallback": stats["fallback"],
        })

    def run_stream(self, output_dir, progress=None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.config.device)
        started = time.perf_counter()
        result = super().run_stream(output_dir, progress=progress)
        if self.reference_mode:
            self.trace.verify_feedback(self.oracle.audit())
            actual_by_step = Counter(row["step"] for row in self.update_rows)
            mismatches = [
                (step, actual_by_step.get(step, 0), expected)
                for step, expected in enumerate(self.trace.old_updates)
                if actual_by_step.get(step, 0) != expected
            ]
            if mismatches:
                raise ValueError(
                    "Per-step supervised update counts differ from reference: "
                    f"{mismatches[:3]}")
            actual_updates = len(self.update_rows)
            if actual_updates != self.trace.total:
                raise ValueError(
                    f"Actual supervised updates {actual_updates} differ from reference "
                    f"{self.trace.total}")
            result["reference_updates"] = self.trace.total
            result["reference_queries"] = len(self.trace.records)
            result["actual_supervised_updates"] = actual_updates
        output = Path(output_dir)
        if self.graph_rows:
            write_csv(output / "graph.csv", self.graph_rows)
        else:
            (output / "graph.csv").write_text(GRAPH_HEADER + "\n", encoding="utf-8")
        if self.expert_event_rows:
            write_csv(output / "expert_feedback.csv", self.expert_event_rows)
        else:
            (output / "expert_feedback.csv").write_text(
                EXPERT_HEADER + "\n", encoding="utf-8")
        costs = {
            "graph_enabled": self.graph_enabled,
            "total_seconds": time.perf_counter() - started,
            "graph_seconds": self.graph_seconds,
            "solve_seconds": self.solve_seconds,
            "training_seconds": self.training_seconds,
            "graph_batches": self.graph_batches,
            "mean_graph_nodes": (self.graph_node_total / self.graph_batches
                                 if self.graph_batches else 0.0),
            "mean_graph_edges": (self.graph_edge_total / self.graph_batches
                                 if self.graph_batches else 0.0),
            "main_encoder_forward_calls": self.main_forward_calls,
            "teacher_encoder_forward_calls": self.teacher_forward_calls,
            "main_model_forward_calls": self.main_model_calls,
            "teacher_model_forward_calls": self.teacher_model_calls,
            "graph_memory_forward_calls": self.memory_forward_calls,
            "graph_memory_forward_images": self.memory_forward_images,
            "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(self.config.device)
                                    / (1024 ** 2) if torch.cuda.is_available() else None),
            "peak_cuda_reserved_mb": (torch.cuda.max_memory_reserved(self.config.device)
                                      / (1024 ** 2) if torch.cuda.is_available() else None),
        }
        (output / "costs.json").write_text(
            json.dumps(costs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result.update(costs)
        result["method"] = "FCRA-G"
        result["base_method"] = "FCRA"
        return result
