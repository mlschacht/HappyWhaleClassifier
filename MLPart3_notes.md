# ML Part 3 Team Notes

Have a group meeting and record your meeting with audio or video (you can record with zoom!)
Start by discussing part 2 of the assignment:
### What approaches you each took to change the ML model in part 2
- Makayla and Ethan tweaked the main 3 parameters
- amanda and daniella added image weights (Resnet18pytorch) using the pretrain weights
- amanda also tried changing a hyperparameter but it did not seem very successful (batch size)
- daniella decreased learning rate
- daniella tried a transformation between training and testing

### What are similarities and differences between your approaches 
- we all seemed to tweak the main 3 parameters (epochs, learning rate, batch size)

### What types of changes improved model performance

- weights are the most helpful
- decreasing the learning rate seemed to help
- increasing epochs also helped
- lowering the resolution oddly improved Makayla's models


### What types of changes made model performance worse

- batch size didn't always help (Amanda I think)
- optimizer change didn't seem to help Amanda or Ethan


### Based on the whole group's work, what would y'all want to do next in terms of model optimization?
- try adding or tweaking image weights (Resnet18pytorch) using the pretrain weights
- try longer epochs
- try lower learning rate
- look for other options on the Happy Whales leaderboard

### Next, look at code and approaches from the Happy Whales competition leaderboardLinks to an external site.
- not sure what might work best here

### Is there any strategy that someone used that you could implement? 
Ex: learning rate value, # epochs, transformation strategy, or more complex changes to the models
- many winning strategies used completely different models and reading about them was a little hard to understand and confusing to determine how to implement into our version of the model. 


### Are there strategies that you are interested in learning about?
- not sure, we weren't sure what best applies to our model outside of what we already have tried.
- Ethan is interested in setting learning rate really really small 10^-16 and would it perform better. 

## Lastly, make a plan for 1 more round of model optimization

### Assign each member something to attempt for training the model. Use your cumulative experience to decide the most important things to test. 
Group members can make as small as changing the value for a parameter or as big as changing the model.
- everyone set the weights
- batch size 48
- resolution: 244x244

epoch: 10, 12, 14, 16
learning rate: 1e-7, 10, 12, 15


Ethan: epoch = 10, learning rate = e-7
Daniella: epoch = 12, learning rate = 1e-10
Amanda: epoch = 14, learning rate = 1e-12
Makayla: epoch = 16, learning rate = 1e-15

zoom meeting link on Makayla's computer: C:\Users\mlsch\OneDrive\Desktop\PD\2026-03-04 15.07.16 ML part 3 Zoom Meeting

All models were run with: pretrained weights, batchsize = 48-50, and resolution = 244x244

### Comparison Table:

| Team member | learning rate | epochs | Train Loss | Validation Accuracy | Validation Loss | Runtime|
| --- | --- | --- | --- |--- | --- | --- | 
| Ethan | 1e-7 | 10 | --- |--- | --- | --- | 
| Daniella | 1e-10 | 12 | --- |--- | --- | --- | 
| Amanda | 1e-12 | 14 | 0.00046733 | **98.5%** | 0.031663 | 42m 9s | 
| Makayla | 1e-15 | 16 | 8.69087 | **0%** | 8.34692 | 54m 24s | 

## Summary About our Attempts and Recommendations Moving Forward (1-3 paragraphs)

The goal was to keep batchsize, resolution, and use the pretrained weights for all of our models. However, it was a bit confusing to determine if we all were really setting up the same "pretrained weights". For example, after mine and Amanda's first attempts, our metrics were vastly different in success. Her model had a validation accuracy of 98% while mine had a validation accuracy of 0%. Daniella said her metrics showed poor model performance like my metrics as well. 