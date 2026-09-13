import pytest
import torch
from scripts.g2_span_diagnostics import restricted_metrics


def test_span_kl_ignores_other_positions_and_matches_forward_kl():
    teacher = torch.zeros(1,3,256,dtype=torch.float64)
    student = teacher.clone(); student[0,0,0] = 12
    y = torch.tensor([[0,1,2]])
    assert restricted_metrics(student,teacher,y,[1,2])['teacher_kl_sum']==pytest.approx(0,abs=1e-12)
    student[0,1,1] = 2
    result = restricted_metrics(student,teacher,y,[1])
    expected = (teacher[0,1].softmax(-1)*(teacher[0,1].log_softmax(-1)-student[0,1].log_softmax(-1))).sum()
    assert result['teacher_kl_sum']==pytest.approx(float(expected),abs=1e-12)
    assert result['bytes']==1
