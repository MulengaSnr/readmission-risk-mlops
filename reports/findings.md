# Findings

_Written for a non-technical reader: what was built, what it showed, and what it does not show._

## In one paragraph
A model that estimates which patients with diabetes may return to hospital within 30 days now runs as an online service. It gives a risk score with three reasons in about a tenth of a second, it can be traced to the exact code and data that built it, and it is rebuilt and checked automatically whenever the code changes. It is a demonstration trained on public US data from 1999–2008. It has not been used on real patients.

## Key findings
1. **The model still works after removing insurance type.** Insurance may stand in for income, so it was taken out. Reviewing the 20% highest-risk discharges reaches 37.9% of readmissions, against 38.5% with insurance included and 20% if patients were chosen at random.
2. **Most readmissions are still missed.** About 62% are not in the top 20%. The score can help decide who to call first, but it cannot be used to decide who does not need support.
3. **Scores mean about the same thing for every group checked.** For each sex, age and race group reported, predicted and actual readmission rates are within about 1 percentage point.
4. **Patients aged 80 and over remain the weakest group.** The model flags 34.8% of their readmissions, against 37.5% for ages 60–79. That gap is too uncertain to call real or to rule out. Four smaller ethnic groups had too few readmissions to judge.
5. **The monitoring caught both problems planted in a six-month simulation,** and named the cause each time: an older patient population in month 5, and a data feed failing in month 6. The four normal months raised no alarms. The same author planted the problems and built the monitoring, so this shows it works on those problems, not how it would handle unexpected ones.
6. **A change in patients is not always a problem for the model.** In the older-population month, the inputs changed but the risk estimates stayed accurate. An automatic retrain would have been the wrong response, so alerts go to a person to investigate.
7. **An early version of the monitoring gave false alarms** because it judged performance on too few months of results. It now waits for three months. The alert thresholds were not changed to make the results look better.

## Recommendations
These apply if an organisation wanted to take this further:
- **Validate on local, recent data first.** Performance in the NHS is unknown.
- **Complete the information-governance work** before any patient data is involved, including a data protection impact assessment.
- **Review the 80+ group specifically** with clinical input, and add a separate check if the gap holds up on local data.
- **Test whether acting on the score helps.** High risk does not mean a readmission can be prevented, so this needs a trial or similar evidence.

## Caveats and limitations
- The data is old, American and from hospital administrative records, and it has no dates, so changes over time could only be simulated.
- Readmissions to other hospitals are probably not recorded, so the true number is likely higher.
- The monitoring has only been run on simulated traffic.
- The online service is a free-tier demo with no login, and no real users rely on it. It must not be given real patient details.

Full figures are in the [model card](../docs/MODEL_CARD.md).
