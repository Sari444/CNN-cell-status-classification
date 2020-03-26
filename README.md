# CNN-cell-status-classification
This code is made for distinguishing between different states of cells. This states are 
The living cells (liv): cells that adhere to the ground, they have a flat shape that changes in time due to motions. The contrast is very low in this state, while the cell nucleus and vesicels are visible. 
The round cells (round): cells detach from the ground and became a round shape. In this state the cells have a high contrast. This state is a transition state thatcame from either living state or cell division and leads to cell division, cell death or living state.
Cell division (div): cells that divide themselves. This state is selected, then a clear seperation between the daughter cells is visible but the daugther cells are still connected and round.
Cell death (dead): in this state the cells turn black inside or get bubbles. The cells stop moving.
The classification is made by an normal CNN with four classes and 3 convolutional layers.
This is the first step for the cell-tracking program. The next steps will be:
1. Implementing a faster R-CNN for object detection
2. Combine the images to gain a cell tracking over a whole video
3. Evaluate from the video time between divisions, number of divisions, relationship between  cells to get a family tree.
