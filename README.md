<br />
<div align="center">
    <h2 align="center">Facts-of-the-Case: Answering Complex Patient Questions</h2>
    <img src="https://img.freepik.com/free-photo/doctor-talking-patient-side-view_23-2149856213.jpg?t=st=1735901369~exp=1735904969~hmac=a76ef278a5dd9d0aec8083fdd3815bc51441c6f4e12b5563809484a7a5c172fb&w=1380" alt="Patient Image Caring Doctor" width="250">
    <br />
    A German benchmark dataset for real-world question-answer pairs consisting of a user's post and an expert answer. The data is extracted from a publicly open German online forum (Lifeline Expertenrat), cleaned, processed, and anonymized.
</div>

## Experiments
This repository contains code to reproduce the experimental evaluation of the corresponding publication. 
It also includes result plots and additional images [here](./images).

## Dataset
### How to access
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/) 

The dataset is licensed under the [Creative Commons license Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
The dataset can be requested for further research and non-commercial use cases by writing an email to:
```
walther@informatik.uni-heidelberg.de
```
Important: 
Please include a motivation and description of your project. If you meet the license conditions, we will hand out the dataset upon request. 

### Samples

|Property|Description|Example|
|---|---|---|
| conversation_id | unique ID (str) | 15113811 |
| user_question | full, anonymized user post (str) | Steht das Haarwuchsmittel Propezia (Finasterid 1mg) im Zusammenhang mit Polyglobulie? Danke für Ihre Antwort. |
| question_published_at | post publication date and time (Unix time) | 1579787700000 |
| category | posted forum (str) | expertenrat,haut-haare-naegel,haarausfall|
| title | post title (str) | Steht Propezia im Zusammenhang mit Polyglobulie? |
|expert_answer | full, anonymized expert answer (str)| Hallo, Finasterid kann in sehr seltenen Fällen Einfluss darauf haben. Ein Bericht aus dem Jahr 2007 gibt aber an, dass eine Polyglobulie dadurch \"behandelt\" wurde, also nicht mehr vorlag. Richtige Zusammenhänge sind hier aber schwer zu untersuchen. Es scheint manchmal nicht erklärbare Reaktionen zu geben. |
| answer_published_at | answer publciation date and time (Unix time) | 1580056200000|

## Attribution
The paper has been accepted at [BTW 2025](https://btw2025.gi.de/). Citation details will follow soon.
