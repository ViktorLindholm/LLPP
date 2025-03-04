//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>
#include <iostream>

using namespace std;

namespace Ped {
	class Twaypoint;

	class Tagent {
	public:
		Tagent(int posX, int posY);
		Tagent(double posX, double posY);

		// Returns the coordinates of the desired position
		int getDesiredX() const { return desiredPositionX; }
		int getDesiredY() const { return desiredPositionY; }

		// Sets the agent's position
		void setX(int newX) 
		{
			if (xPointer == NULL) { x = newX; }
			else { *xPointer = newX;  }		
		}

		void setY(int newY)
		{
			if (yPointer == NULL) { y = newY; }
			else { *yPointer = newY; }
		}

		void setXPointer(float *newP) { xPointer = newP; }
		void setYPointer(float *newP) { yPointer = newP; }
		void setXDestinationPointer(float *newP) { xDestinationPointer = newP; }
		void setYDestinationPointer(float *newP) { yDestinationPointer = newP; }
		void setRDestinationPointer(float *newP) { rDestinationPointer = newP; }

		Twaypoint* getDestination() {
			return destination; 
		}
		void setDestination(Twaypoint *dst) {
			this->destination = dst;
		}

		// Update the position according to get closer
		// to the current destination
		void computeNextDesiredPosition();
		void computeNextDesiredPositionVECTOR();

		// Position of agent defined by x and y
		int getX() const 
		{ 
			if (xPointer == NULL) { return x; }
			else { return (*xPointer); }
		};


		int getY() const 
		{ 
			if (yPointer == NULL) { return y; }
			else { return (*yPointer); }
		};

		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint* wp);

		// Returns the next destination to visit
		Twaypoint* getNextDestination();

		Twaypoint* getNextDestinationVector(float precalc_value);

	private:
		Tagent() {};

		// The agent's current position
		int x;
		int y;
		float *xPointer = NULL;
		float *yPointer = NULL;
		float *xDestinationPointer = NULL;
		float *yDestinationPointer = NULL;
		float *rDestinationPointer = NULL;

		// The agent's desired next position
		int desiredPositionX;
		int desiredPositionY;

		// The current destination (may require several steps to reach)
		Twaypoint* destination;

		// The last destination
		Twaypoint* lastDestination;

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint*> waypoints;

		// Internal init function 
		void init(int posX, int posY);


	};
}

#endif