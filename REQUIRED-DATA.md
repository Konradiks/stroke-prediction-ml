# Required data types accepted by model

A list of features used by the model to assess stroke risk, including medical, demographic, and lifestyle information.

---

## Integer values

**Age** – Integer value between 1 and 100 (validation to be confirmed).

**Systolic BP** – The pressure in arteries when the heart beats (contracts), as an unsigned integer.

**Diastolic BP** – The pressure in arteries between beats (relaxation phase), as an unsigned integer.

**HDL Cholesterol** – "Good" cholesterol value in mg/dL, as an unsigned integer.

**LDL Cholesterol** – "Bad" cholesterol value in mg/dL, as an unsigned integer.

---

## Float values

**Stress Levels** – Perceived stress level from 0.00 to 9.99, with a step of 0.01.

**Average Glucose Level** – Blood glucose concentration in mg/dL, from 60.00 to 200.00, with a step of 0.01.

**Body Mass Index (BMI)** – Body fat estimate based on height and weight, from 15.01 to 40.00, with a step of 0.01.

---

## Binary representation

**Gender** – Binary: 0 for female, 1 for male.

**Residence Type** – Binary: 0 for Urban, 1 for Rural.

**Family History of Stroke** – Binary: 0 for No, 1 for Yes.

---

## Boolean values as int (0 - false, 1 - true)

### True or False

**Hypertension** – Whether the patient has high blood pressure.

**Heart Disease** – Whether the patient has any diagnosed heart condition.

**Stroke History** – Whether the patient has had a stroke in the past.

---

## Pick only one

### Marital Status

**Married** – Patient is currently married.

**Single** – Patient has never been married.

**Divorced** – Patient is legally divorced.

---

### Work Type

**Self-employed** – Runs own business or works independently.

**Never Worked** – No work history.

**Private** – Employed in the private sector.

**Government Job** – Employed in a public/government sector.

---

### Smoking Status

**Non-smoker** – Has never smoked.
**Formerly Smoked** – Smoked in the past but not currently.

**Currently Smokes** – Actively smokes cigarettes or tobacco.

---

### Alcohol Intake

**Social Drinker** – Drinks occasionally in social settings.

**Never** – Has never consumed alcohol.

**Rarely** – Drinks infrequently.

**Frequent Drinker** – Consumes alcohol regularly.

---

### Physical Activity

**Moderate** – Engages in moderate physical exercise regularly.

**Low** – Performs little to no physical activity.

**High** – Frequently participates in intense physical activity.

---

### Dietary Habits

**Vegan** – Excludes all animal products.

**Paleo** – Follows a diet based on Paleolithic-era foods.

**Pescatarian** – Vegetarian diet that includes fish.

**Gluten-Free** – Avoids all gluten-containing foods.

**Vegetarian** – Excludes meat but may include dairy/eggs.

**Non-Vegetarian** – Regularly consumes meat and animal products.

**Keto** – High-fat, low-carb ketogenic diet.

---

## Symptoms – multi-pick (multiple symptoms can be selected)

**Difficulty Speaking** – Trouble forming or articulating words.

**Headache** – Persistent or severe headache.

**Loss of Balance** – Trouble maintaining physical balance.

**Dizziness** – Feeling lightheaded or faint.

**Confusion** – Disoriented or trouble understanding.

**Seizures** – Uncontrolled electrical disturbances in the brain.

**Blurred Vision** – Difficulty seeing clearly.

**Severe Fatigue** – Extreme tiredness or lack of energy.

**Numbness** – Loss of sensation in part of the body.

**Weakness** – Reduced physical strength, typically on one side.

