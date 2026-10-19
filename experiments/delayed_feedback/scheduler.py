"""Dispatch independent jobs, keeping at most one worker on each GPU."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def run_jobs(jobs, gpus, launch, on_result):
    """Refill whichever GPU finishes first; only the caller writes result tables.

    launch(job, gpu) executes one isolated worker and returns its result.
    On error, stop submitting jobs and allow already running workers to finish.
    """
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide distinct nonnegative GPU indices")
    jobs = iter(jobs)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        pending = {}

        def submit(gpu):
            job = next(jobs, None)
            if job is not None:
                pending[pool.submit(launch, job, gpu)] = gpu

        for gpu in gpus:
            submit(gpu)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            completed = []
            failure = None
            for future in done:
                gpu = pending.pop(future)
                try:
                    completed.append((gpu, future.result()))
                except Exception as error:
                    failure = error
            for gpu, result in completed:
                on_result(result)
            if failure is not None:
                # The context manager waits for the remaining active worker(s).
                # Preserve their successful results too, without launching more.
                for future in pending:
                    try:
                        result = future.result()
                    except Exception:
                        continue
                    on_result(result)
                raise failure
            for gpu, _ in completed:
                submit(gpu)
