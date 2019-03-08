//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_waypoint.h"
#include <math.h>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

Ped::Tagent::Tagent(int posX, int posY) {
	Ped::Tagent::init(posX, posY);
}

Ped::Tagent::Tagent(double posX, double posY) {
	Ped::Tagent::init((int)round(posX), (int)round(posY));
}

void Ped::Tagent::init(int posX, int posY) {
	x = posX;
	y = posY;
	destination = NULL;
	lastDestination = NULL;
}

void Ped::Tagent::computeNextDesiredPositionVECTOR() {
	if (this->destination == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}

	double diffX = destination->getx() - this->getX();
	double diffY = destination->gety() - this->getY();
	double len = sqrt(diffX * diffX + diffY * diffY);
	desiredPositionX = (int)round(this->getX() + diffX / len);
	desiredPositionY = (int)round(this->getY() + diffY / len);
}

void Ped::Tagent::computeNextDesiredPosition() {
	destination = getNextDestination();
	if (destination == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}

	double diffX = destination->getx() - this->getX();
	double diffY = destination->gety() - this->getY();
	double len = sqrt(diffX * diffX + diffY * diffY);
	desiredPositionX = (int)round(this->getX() + diffX / len);
	desiredPositionY = (int)round(this->getY() + diffY / len);
}

void Ped::Tagent::addWaypoint(Twaypoint* wp) {
	waypoints.push_back(wp);
}

Ped::Twaypoint* Ped::Tagent::getNextDestination() {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (destination != NULL) {
		// compute if agent reached its current destination
		double diffX = destination->getx() - this->getX();
		double diffY = destination->gety() - this->getY();
		double length = sqrt(diffX * diffX + diffY * diffY);

		agentReachedDestination = length < destination->getr();
	}

	if ((agentReachedDestination || destination == NULL) && !waypoints.empty()) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		waypoints.push_back(destination);
		nextDestination = waypoints.front();
		waypoints.pop_front();	
		
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = destination;
	}

	if (xDestinationPointer != NULL)
	{
		if (nextDestination != NULL)
		{
			*this->xDestinationPointer = nextDestination->getx();
			*this->yDestinationPointer = nextDestination->gety();
			*this->rDestinationPointer = nextDestination->getr();
		}
	}

	return nextDestination;
}

Ped::Twaypoint* Ped::Tagent::getNextDestinationVector(float length) {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (destination != NULL) {
		agentReachedDestination = length < destination->getr();
	}

	if ((agentReachedDestination || destination == NULL) && !waypoints.empty()) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		waypoints.push_back(destination);
		nextDestination = waypoints.front();
		waypoints.pop_front();

	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = destination;
	}

	if (xDestinationPointer != NULL)
	{
		if (nextDestination != NULL)
		{
			*this->xDestinationPointer = nextDestination->getx();
			*this->yDestinationPointer = nextDestination->gety();
			*this->rDestinationPointer = nextDestination->getr();
		}
	}

	return nextDestination;
}
