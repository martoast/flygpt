# Compiler v3 independent replication

Preregistered five-seed replication conditional on a fixed teacher, task and training corpus; small-N inference

All 30 models completed on the M4 before final-test evaluation. The pilot is excluded.

| Seed | Method | Exact % | Response CE | Zero-edge exact % |
|---|---|---:|---:|---:|
| 400 | A_hard | 89.062 | 0.056676 | 0.000 |
| 400 | B_soft | 95.703 | 0.020976 | 0.000 |
| 400 | C_hidden | 100.000 | 0.004615 | 0.000 |
| 400 | D_hidden_shuffled | 100.000 | 0.008122 | 0.000 |
| 400 | E_relational | 98.828 | 0.012233 | 0.000 |
| 400 | F_relational_shuffled | 81.445 | 0.075113 | 0.000 |
| 401 | A_hard | 98.438 | 0.019129 | 0.000 |
| 401 | B_soft | 98.242 | 0.007960 | 0.000 |
| 401 | C_hidden | 100.000 | 0.003937 | 0.000 |
| 401 | D_hidden_shuffled | 98.828 | 0.015952 | 0.000 |
| 401 | E_relational | 98.047 | 0.016445 | 0.000 |
| 401 | F_relational_shuffled | 82.812 | 0.069953 | 0.000 |
| 402 | A_hard | 91.602 | 0.045213 | 0.000 |
| 402 | B_soft | 99.023 | 0.003617 | 0.000 |
| 402 | C_hidden | 100.000 | 0.007240 | 0.000 |
| 402 | D_hidden_shuffled | 96.484 | 0.027181 | 0.000 |
| 402 | E_relational | 99.023 | 0.012556 | 0.000 |
| 402 | F_relational_shuffled | 99.609 | 0.008959 | 0.000 |
| 403 | A_hard | 100.000 | 0.006204 | 0.000 |
| 403 | B_soft | 95.117 | 0.028234 | 0.000 |
| 403 | C_hidden | 100.000 | 0.005356 | 0.000 |
| 403 | D_hidden_shuffled | 99.219 | 0.011701 | 0.000 |
| 403 | E_relational | 99.805 | 0.006645 | 0.000 |
| 403 | F_relational_shuffled | 98.242 | 0.015992 | 0.000 |
| 404 | A_hard | 97.852 | 0.022017 | 0.000 |
| 404 | B_soft | 99.805 | 0.002337 | 0.000 |
| 404 | C_hidden | 100.000 | 0.006026 | 0.000 |
| 404 | D_hidden_shuffled | 91.406 | 0.046660 | 0.000 |
| 404 | E_relational | 98.242 | 0.016915 | 0.000 |
| 404 | F_relational_shuffled | 99.609 | 0.012997 | 0.000 |

Positive benefit means higher accuracy or lower CE. Accuracy benefits below are fractions, not percentage points.

| Method vs control | Endpoint | Mean benefit | 95% t interval (unadjusted) | Holm p | Exact sign-flip p | Meets registered t criterion |
|---|---|---:|---|---:|---:|---|
| B_soft vs A_hard | accuracy | 0.021875 | [-0.04111759260767484, 0.08486759260767485] | 0.95481 | 0.375 | False |
| B_soft vs A_hard | response_ce | 0.017223 | [-0.013935914232471917, 0.048381699490524656] | 0.95481 | 0.25 | False |
| C_hidden vs D_hidden_shuffled | accuracy | 0.028125 | [-0.015175537202863295, 0.07142553720286329] | 0.95481 | 0.125 | False |
| C_hidden vs D_hidden_shuffled | response_ce | 0.016488 | [-0.0019947092911329414, 0.03497161589321902] | 0.616 | 0.0625 | False |
| E_relational vs F_relational_shuffled | accuracy | 0.064453 | [-0.04853011457607974, 0.17743636457607975] | 0.95481 | 0.3125 | False |
| E_relational vs F_relational_shuffled | response_ce | 0.023644 | [-0.01629023601429309, 0.06357746407810695] | 0.95481 | 0.25 | False |
| C_hidden vs A_hard | accuracy | 0.046094 | [-0.01312599496353007, 0.10531349496353007] | 0.7742 | 0.125 | False |
| C_hidden vs A_hard | response_ce | 0.024413 | [-0.0008786927243448414, 0.04970443062085098] | 0.55229 | 0.0625 | False |
| E_relational vs A_hard | accuracy | 0.033984 | [-0.025901974157397853, 0.09387072415739785] | 0.95481 | 0.375 | False |
| E_relational vs A_hard | response_ce | 0.016889 | [-0.008321379226496551, 0.04209874085594152] | 0.95481 | 0.125 | False |

Five paired seeds; t assumptions uncertain; exact two-sided sign-flip minimum p=.0625; fixed teacher and corpus
Representation benefit requires both shuffled and hard-answer contrasts on the same endpoint. No pooled pilot inference.
Complete curves, threshold intervals/censoring and standalone acquisition/end-to-end timing are in learning.json. Final-test evaluation times are separate. No update-efficiency claim follows from coincident observed threshold crossings.
