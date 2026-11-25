# Limitations and ethical considerations

## Forecasting
- **Short history.** The age breakdown starts in April 2022 (53 months), and the
  backtest has 18 origins. Accuracy figures are useful but have wide uncertainty.
- **Referrals are not need.** Referrals reflect awareness, school and GP referral
  behaviour, and service availability (including Mental Health Support Teams in
  schools). A policy change, such as a new referral route or a change in recording,
  would break the patterns these models learn.
- **Provisional data.** The last 5 months are Performance data and may be revised
  upward as late submissions arrive. That would make the recent level look slightly
  low.

## Simulator
- **No demonstrated predictive skill.** On the later year it is no better than assuming
  no change for median wait (3.4% vs 3.5%) and worse for the 90th percentile (12.1% vs
  5.4%). The list-size match is partly circular, and the model was revised after looking
  at that year. Treat every scenario as an illustration of the assumptions.
- **One national queue.** In reality there are hundreds of provider waiting lists
  with different pathways (neurodevelopmental, eating disorders, crisis). National
  aggregates hide local variation; see Project 1 for the geography.
- **Priority is inferred, not observed.** The "newest first" rule is the simplest
  rule that reproduces the published median and 90th-percentile waits. It
  describes how the list behaves *as a whole*. It is not a claim about how any
  clinician triages. Much of it likely reflects pathway mix: short "other mental
  health" waits next to very long neurodevelopmental waits.
- **The calibrated priority is weakly identified.** The data fit equally well for
  any strong priority (k_serve ≥ 0.1/day). The scenarios use a parsimonious value.
  A stronger priority would make the "long-waiters left behind" effect slightly
  larger, not smaller.
- **First contact is not treatment.** The model stops the clock at a first contact,
  not at the second contact or full "meaningful help" (MRS08).
- **FIFO is a mathematical bound, not a recommendation.** Clinically urgent
  children must always be seen first. The ring-fenced policy is a less
  extreme illustration: part of capacity reserved for long-waiters, with clinical prioritisation
  for the rest.
- **The scenarios hold behaviour fixed.** They assume extra capacity doesn't change
  referral rates (induced demand) or exit rates.

## Ethics
- Aggregate open data only. No child can be identified.
- The scenarios are presented as illustrative trade-offs, not capacity plans.
  Waiting-list decisions affect vulnerable children and need clinical and
  operational input that a statistical model cannot provide.
- Terminology follows NHS England's "children and young people (CYP)" wording.
