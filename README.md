# LLPP

Crowd (or Pedestrian) simulation is the study of the movement of people through
various environments in order to understand the collective behaviour of crowds in
different situations. Accurate simulations could be useful when designing airports or
other public spaces, for understanding crowd behaviour departing sporting events or
festivals, and for dealing with emergencies related to such situations. Understanding
crowd behaviour in normal and abnormal situations helps improve the design of the
public spaces for both efficiency of movement and safety, answering questions such as
where the (emergency) exits should be put.
This system simulates an area of people walking around. Each person, or agent,
is moving towards a circular sequence of waypoints
, in the 2D area. The starting
locations and waypoints are specified in a scenario configuration file
, which is loaded at the beginning of the program run.
