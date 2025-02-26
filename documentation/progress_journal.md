# Team 7 Progress Journal - Rough Draft

## Team Meeting Notes

### Meeting 1 - January 20th

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

### Meeting 2 - January 27th

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
 
### Meeting 3 - February 7th

- Literature review is due in 2 weeks
- Literature Review Progress:
  - Identified key Correlation Filter SOT techniques for our solution
  - We narrowed down on five Correlation Filter Techniques, and each member of group focussed on diving into their selected SOT strategy
  - Gathered Source code for each of these SOT techniques. We will start looking at implementing these models over the next few weeks
 
#### For next week, each team member must:

- Refine selected model's source code, begin looking at implementing a working model
- Begin working on the literature review document, gather and present sources used in the 'Related Works' section of the document
- Analyse the strengths and weaknesses of their selected Correlation Filter based SOT technique

#### In next week's meeting, we should:

- Discuss how the team is planning on completing the literature review document
  - Allocate specific tasks to group members for completing the literature review
- Work on setting up a functional testing environment
- Complete a rough implementation of the SOT model.

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
