# Team 7 Progress Journal - Team Meeting Notes

## Meeting 1 - January 20th

#### Discussion Points

- We will firstly work towards the literature review from part 1 of the project
- We have 4 weeks of work until this is due

#### For next week, each team member must:

- Read 2 or more recent research paper on Single object tracking task, at least one of which should be using traditional computer vision methods (that means it is not using deep leaning approach)
- Prepare a quick report or summary of the research to discuss in next meeting
  - Each team member should have a good understanding of the foundations of their papers, so that we can share with each other to get a better understanding of single object tracking techniques
- Learn about pytorch if they do not have previous background knowledge

#### In next week's meeting, we should:

- Share our findings
- Aim to hone our literature review efforts into the papers that will help us define our problem statement for part 2 of the project
- Put together a Kanban or progress board where we can track to-dos. This will help with the development in the repo for part 2

## Meeting 2 - January 27th

#### Discussion Points

- Literature review is due in 3 weeks
- Discussion on our research papers:
  - There are many SOT techniques still being researched to address issues with variations in occlusion, lighting (poor conditions or rapid changes) scale, pose, and deformations.
  - Many traditional correlation filters have been used to address these issues, yet they still fall short in many scenarios
    - We should to try to borrow various different advancements that build upon this technique to build our own for a specific task
    - Which task will we focus on?
  - Emerging deep learning models can address these challenges and more:
    - We should aim to reduce computation complexity of the model to increase efficiency so that it can be used on-device in more applications
    - We could build a pipeline to combine traditional methods (e.g. correlation filters) with these newer learning methods. Is this still considered a traditional method to prof?

#### For next week, each team member must:

- Continue researching your paper more in depth, and bring a technical breakdown of one or two models/approaches to our meeting
  - Prioritize papers with available source code, so that we can be prepared for implementation next week
- Describe the current issues in SOT (Single Object Tracking) that you have seen in your papers using a couple sentences

#### In next week's meeting, we should:

- Create a single problem statement using each of our individual parts
- Try to replicate at least something from one paper using cited source code based on our new technical understanding
- Build a Kanban board AND create to-do's that will keep us on track for the literature review

  - Implementation
  - Consolidating findings
  - Pros and Cons for each paper

## Meeting 3 - February 3rd

#### Discussion Points

- We have 2 meetings until the literature review is due
  - We need to focus our efforts towards the specific requirements for it
- There are many different computer vision models being worked on currently for single object tracking, and through our research we have found different methods are most useful in specific scenarios. Because of this, we have determined we will work towards creating an ensemble model using the top picks from our research thus far.
  - Ensemble models see success in other applications of ML, so we see no reason why they wouldn't be effective in this case
  - We will select 5 Correlation Filter-Based SoTA Models (1 per group member), and begin our effort to construct an ensemble model based on these
    - We have compiled a list of these models using our research thus far

#### For next week, each team member must:

- Select a Correlation Filter-Based SoTA Model (list in Discord) and find a paper introducing or utilizing it which **includes source code**
  - **Riley**: SRDCF (Spatially Regularized Discriminative Correlation Filter)
  - **Wanju**: KCF (Kernelized Correlation Filter)
  - **Henry**: CSK?
  - **Dichen**: MOSSE
  - **Santam**: ?
- Understand the nuances of your selected model. Then, put together a list of scenarios where it performs best, as well as anything it struggles with. This will help us plan and construct the ensemble model.
- Attempt to reproduce the results of a paper on your model using the provided source code.

#### In next week's meeting, we should:

- Create a single problem statement, using the problems that our separate models aim to solve
- Begin planning how the ensemble model will be structured based on the advantages and disadvantages of each model. Will we use all 5 or cut it down to 3 or 4 models?
- Break the literature review down into pieces that we can each individually contribute to
  - Then, create to-do's for each member to keep us on track

### Meeting 4 - February 14th

- Literature review is due in a week
- Consolidated findings of each group member and each SOT method
- Troubleshooting source code setup issues, making progress
- Literature Review Progress:
  - We have enough information and sources to effectively start working on completing the upcoming literature review deliverable

#### For next week, each team member must:

- Continue drafting and organizing the literature review
- Finalize the problem statement
- Ensure at least one source code implementation runs properly

#### In next week's meeting, we should:

- Ensure the team is on track to meet the deadline for the literature review submission
- Go over source code implementation and try to get at least one of them working
- Decide which image dataset the team is going to use for evaluating models, to ensure consistency

### Meeting 5 - February 21st

- Successfully ran one team member's code implementation
- Selected the OTB2015 and LaSOT datasets for initial benchmarking
- Worked on completing the literature review report
- Assigned tasks to group members for completing the upcoming assignment

#### For next week, each team member must:

- Until the deadline(Monday), finalize report sections and ensure a cohesive write up
- Run additional tests with the chosen dataset
- Check citations and references in the document
- Add LaTex formatting to achieve a visually pleasing document

#### In next week's meeting, we should:

- Focus on getting more team member's models up and running
- Run experiments on the chosen dataset and gather performance metrics to compare efficiency across selected methods
- Define goals for the next phase of development

### Meeting 6 - February 28th

- Now that we have submitted our report, we need to start getting more code working
- We will be meeting with the TA today, lets make sure we know what to discuss. Have any questions ready

#### For next week, each team member must:

- Complete the replication of their model
- Upload their working code to their respective sub-folder in the repository under `src/MODEL_NAME`
- Work on getting their model benchmarking on OTB2015

#### In next week's meeting, we should:

- Discuss any problems we have run into when replicating
- Break down the problem of how to ensemble all the models together

### Meeting 7 - March 6th

- Focused on replicating individual tracking models
- Discussed model integration and file formatting for ensemble testing 
- Decide on file naming convention and folder structure for organizing the repository
- Begin work on a tradional SOT model to fulfill project requirements

#### For next week, each team member must:

- Push clean, working code to the repository
- Convert outputs to standardized format
- Benchmark their model on OTB2015 sequences

#### In next week's meeting, we should:

- Finalize the format for tracker outputs
- Begin writing wrapper for ensemble logic

### Meeting 8 - March 13th

- Initial benchmarks from OTB2015 sequences collected
- Started working on a MATLAB wrapper to unify outputs across trackers
- Identified differences in bounding box formats and output styles between models
- Discussed model integration and file formatting for ensemble testing

#### For next week, each team member must:

- Validate that their tracker outputs are aligned and compatible with ensemble wrapper
- Begin visualizing tracker overlap

#### In next week's meeting, we should:

- Troubleshoot any alignment or output issues

  ### Meeting 9 - March 20th

- Added visualizations to display all individual bounding boxes and the ensemble result
- KCF (Python/OpenCV) and STRCF (C/MEX + OpenCV) running; DSST and MOSSE working in MATLAB but still being debugged
- Established Benchmarks and Accuracy metrics across different models

#### For next week, each team member must:

- Start preparing figures and tables for final report
- Look into developing a solution for the live demonstration
- Begin working on the final project deliverables, including the presentation and the final report

#### In next week's meeting, we should:

- Determine a strategy for completing the project deliverables and handing in finished code

  ### Meeting 10 - March 27th

- System more stable now on difficult sequences
- Drafted content for Final Report
- STRCF successfully tested on live webcam input

#### For next week, each team member must:

- Finish system integrations and consolidate different SOT models
- Finalize model performance plots and benchmark tables
- Polish code for submission (remove test scripts, comment code)
- Finalize all documentation and assign final tasks before submission
- Plan the presentation and live demonstration
