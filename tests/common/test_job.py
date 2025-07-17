
from pathlib import Path
from .conftest import *

def test_job_run(job):
    assert not job.completed

    iteration_num = 0

    first_ts = None

    for iteration in job.run():
        assert iteration_num == len(job.state.iterations) - 1

        if first_ts is None:
            first_ts = iteration.timestamp
        else:
            assert iteration.timestamp > first_ts
        
        iteration_num += 1

    job.save_state()

    assert job.completed

    # check that the state file exists
    assert Path(job.state_location).exists()

