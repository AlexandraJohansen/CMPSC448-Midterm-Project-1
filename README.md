# CMPSC448-Midterm-Project-1
Who Wrote It? Identifying LLMs from Their
Responses
(CMPSC 448; modified 9/17/2026)
Deadline: 11:59pm on Oct. 11, 2026

Project Overview: 
As Large Language Models (LLMs) such as GPT, Claude,

Gemini, Llama, and Qwen increasingly generate the text we read online, we

face an emerging digital authorship crisis: Can we still tell who—or
what—wrote a piece of text? Different LLMs may leave distinct linguistic,
stylistic, and structural “fingerprints” in their responses, raising an intriguing
question: Can machine learning identify the AI behind the text?
In this project, each team can obtain or construct data with at least three
components: “(LLM_name, LLM_Input, LLM_output)”. Your dataset should
contain responses from at least 3 different LLM families. Examples include,
but are not limited to: GPT, Claude, Gemini, Llama, Qwen, DeepSeek, other
LLMs. You may obtain data from: publicly released datasets; publicly available
LLM outputs; LLM APIs or interfaces; interactions that you generate yourself;
your own LLM interactions that you are comfortable using for this project. You
should clearly document where your data came from and how it was collected
or constructed. Do not include private, sensitive, confidential,
copyrighted-inappropriately-reproduced, or third-party information that you do
not have the right to use. Try to keep the number of examples from different
LLMs reasonably balanced.
Teams: You may work individually or in a team of up to 5 students. Each
team should choose a team leader who is responsible for:
● leading and coordinating the project;
● submitting the project;
Required Models: You must implement and evaluate the following two
models: Convolutional Neural Network (CNN) and Recurrent Neural Network
(RNN, like LSTM). You may additionally experiment with other classifiers or
combine models to obtain better performance. However, CNN and RNN are
required.
Research Questions (each team must investigate at least RQ1 and RQ2.
You are encouraged to further investigate RQ3 and/or RQ4 for extra
credit. Each additional RQ is worth 5 extra-credit points.).
● RQ1 — Can we identify which LLM generated a response?
Use LLM_output as input, conduct classification to detect which LLM
generated it. Report the performance of your CNN&RNN performance
● RQ2 — Does the user's prompt help identify the LLM?
Compare compare classifiers using: Input only; Output only; Input +
Output
Questions you may consider: i) Can the LLM be predicted from the
user's prompt even before seeing its response? ii) Does including the
input improve classification? iii) If input-only classification works
surprisingly well, why might this happen?
● RQ3 — Do LLM fingerprints generalize across tasks or domains?
(Extra credit “5”)
Investigate whether a classifier trained on one type of task can identify
LLMs on another type of task.
For example: Training on problems like coding + mathematics + factual
questions, but Testing on writing. Or Training on general question
answering but Testing on coding.
Questions you may consider: i) Does performance decrease on unseen
tasks? ii) Are some models more consistently identifiable across
domains? iii) Is the classifier learning a general LLM fingerprint or
task-specific patterns?
● RQ4 — What characteristics distinguish different LLMs? (Extra credit
“5”)
Classification accuracy tells us whether models can be distinguished.
This RQ asks why. Analyze linguistic or structural characteristics such
as: response length; vocabulary; lexical diversity; formatting, etc.
Questions you may consider: i) What signals does the classifier appear
to use? ii) Are the signals linguistic, structural, or both? iii) Does
removing certain signals substantially reduce classification
performance?
What You Can Use
● PyTorch; NumPy; pandas; scikit-learn; NLTK; spaCy; other standard
Python/ML packages;
● publicly available datasets.
● You may use LLMs such as ChatGPT, Claude, Gemini, etc. to help you:
understand concepts; debug code; brainstorm experiments; interpret
results. However, you are responsible for understanding, verifying, and
explaining everything submitted by your team.
What You Need to Submit
● Submit the URL of your GitHub repository. Your repository should
include:
○ Code

○ Project Report in pdf (names of team leader and members
included)
Evaluation:
● PDF quality (80%): we score by the following five aspects:
○ Problem definition and Dataset curation (20 points)

○ Your CNN/RNN implementation and how you train it (i.e., your
model details and training details) (20 points)

○ The results and how they are presented (20 points)

○ Any in-depth analyses/experiments (20 points)

○ Lessons&Experience you learned in this project (20 points)
For individual participants, points will be deducted in units of 5; for
group participants, points will be deducted in units of 4+2*G where “G” refers
to group size (2,3,4 or 5).

● (optional) Presentation (20%): Each participant (individual or group)
can choose whether you want to present or not. You can get a default
score 17 (out of max 20) for the presentation part if you choose NOT
(such as your scores of other parts can secure an “A” for you). If you
choose to present, the evaluation criteria include: slide quality, work
quality, presentation skills, question answering, etc.) For individual
presentations, points will be deducted in units of 2; for group
presentations, points will be deducted in units of 4.

● If it is a group project, each team member gets the same score
except the team leader has an extra credit score “3”
