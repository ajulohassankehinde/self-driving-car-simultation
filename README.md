# self-driving-car-simultation

This project is intended to simulate self driving car model however real cars have much more complex algorithms, decision making processes and deep learning concepts. This is a simplified project of the same.
Detials of the project are : 
We create a enviornment using kivy library.
We build a simple car model with 3 sensors upfront. These sensors will detect any kind of obstackle infront of it. It has a certain radius in which it tries to detect the obstacle(20 units).
We then use Q learning, also referred as quality learning, which is the simplest reinforcement learning algorithm. 
we draw obstackles on the enviroment for the car and then set GOAL A and GOAL B for the car and the car keeps travelling back and forth from these goals. The cars tries to learn a path and update its learning if any obstackle comes in between.
The car is penalized for wrong path.
