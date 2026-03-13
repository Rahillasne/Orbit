# Post-Launch Checklist — ORBIT v1.2.0

## Week 1 After Launch

- [ ] Respond to every GitHub issue within 4 hours
- [ ] Respond to every Discord/Reddit/Twitter comment within 6 hours
- [ ] Track: how many pip installs (check PyPI stats after 48h)
- [ ] Track: how many GitHub stars
- [ ] Ask anyone who tries it: "what was confusing?" and "what would make this useful for you?"

## Ongoing (Path to rho = 0.70)

- [ ] Fix R3M weights (find actual weights URL, fix SSL certs) — estimated +0.05 rho
- [ ] Test extended features (9 new temporal/geometry features) — needs raw re-profiling
- [ ] Every new community dataset with known success rate: add to ground truth, re-profile, retrain
- [ ] Target: 60+ real profiled datasets by end of month
- [ ] When rho crosses 0.70: write a proper paper, submit to RSS or CoRL workshop

## Community Building

- [ ] Open a GitHub Discussion for "share your dataset results"
- [ ] Create an issue template for "add new dataset to ground truth"
- [ ] Write a CONTRIBUTING.md explaining how to profile a dataset and submit results
