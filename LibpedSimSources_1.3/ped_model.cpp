//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include <cuda_runtime.h>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <assert.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
	//assert(false);
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();

	// Set up heatmapCUDA
	setupHeatmapCUDA();

	// ##########################################

	this->section1 = (section_t *)calloc(1, sizeof(section_t));
	this->section2 = (section_t *)calloc(1, sizeof(section_t));
	this->section3 = (section_t *)calloc(1, sizeof(section_t));
	this->section4 = (section_t *)calloc(1, sizeof(section_t));
	setup_section(this->section1, 0, 40, NULL, this->section2);
	setup_section(this->section2, 40, 80, this->section1, this->section3);
	setup_section(this->section3, 80, 120, this->section2, this->section4);
	setup_section(this->section4, 120, 160, this->section3, NULL);

	for (Tagent *Agent : agentsInScenario)
	{
		int X = Agent->getX();
		if (X < 40)
		{
			this->section1->agents_in_section.push_back(Agent);
		}
		else if (X < 80)
		{
			this->section2->agents_in_section.push_back(Agent);
		}
		else if (X < 120)
		{
			this->section3->agents_in_section.push_back(Agent);
		}
		else if (X <= 160)
		{
			this->section4->agents_in_section.push_back(Agent);
		}
	}

	// ##########################################

	if (implementation == VECTOR)
	{ 
		this->arrayX = (float *)_mm_malloc(agentsInScenario.size() * sizeof(float), 16);
		this->arrayY = (float *)_mm_malloc(agentsInScenario.size() * sizeof(float), 16);
		this->destinationX = (float *)_mm_malloc(agentsInScenario.size() * sizeof(float), 16);
		this->destinationY = (float *)_mm_malloc(agentsInScenario.size() * sizeof(float), 16);
		this->destinationR = (float *)_mm_malloc(agentsInScenario.size() * sizeof(float), 16);
		float *currentX = arrayX;
		float *currentY = arrayY;
		float *currentXDestination = destinationX;
		float *currentYDestination = destinationY;
		float *currentRDestination = destinationR;

		for each (Ped::Tagent* Agent in agentsInScenario)
		{
			*currentX = Agent->getX();
			*currentY = Agent->getY();

			Agent->setXPointer(currentX);
			Agent->setYPointer(currentY);
			Agent->setXDestinationPointer(currentXDestination);
			Agent->setYDestinationPointer(currentYDestination);
			Agent->setRDestinationPointer(currentRDestination);

			currentX++;
			currentY++;
			currentXDestination++;
			currentYDestination++;
			currentRDestination++;
		}
	}
}

void tickVECTOR(std::vector<Ped::Tagent*> allAgents, float *arrayX, float *arrayY, float *destinationX, float *destinationY, float *destinationR)
{
	
	float *preCalcArray = (float *)_mm_malloc(allAgents.size() * sizeof(float), 16);

	__m128 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;
	for (int i = 0; i < allAgents.size(); i+=4)
	{
		t0 = _mm_load_ps(&destinationX[i]);
		t1 = _mm_load_ps(&arrayX[i]);
		t2 = _mm_sub_ps(t0, t1);

		t3 = _mm_load_ps(&destinationY[i]);
		t4 = _mm_load_ps(&arrayY[i]);
		t5 = _mm_sub_ps(t3, t4);

		t6 = _mm_mul_ps(t2, t2);
		t7 = _mm_mul_ps(t5, t5);
		t8 = _mm_add_ps(t6, t7);
		t8 = _mm_sqrt_ps(t8);

		_mm_store_ps(&preCalcArray[i], t8);

		for (size_t j = i; j < i+4; j++)
		{
			Ped::Tagent *Agent = allAgents[j];
			Agent->setDestination(Agent->getNextDestinationVector(preCalcArray[j]));
		}
		
		t2 = _mm_div_ps(t2, t8);
		t2 = _mm_add_ps(t1, t2);
		t2 = _mm_round_ps(t2, _MM_FROUND_TO_NEAREST_INT);
		_mm_store_ps(&arrayX[i], t2);
		
		t5 = _mm_div_ps(t5, t8);
		t5 = _mm_add_ps(t4, t5);
		t5 = _mm_round_ps(t5, _MM_FROUND_TO_NEAREST_INT);
		_mm_store_ps(&arrayY[i], t5);
	}
	
}

void Ped::Model::tickSEQ(std::vector<Ped::Tagent*> allAgents)
{
	for each (Ped::Tagent* Agent in allAgents)
	{
		Agent->computeNextDesiredPosition();
		this->move(Agent);
	}
	updateHeatmapSeq();
}

void tickPTHREADAux(std::vector<Ped::Tagent*> allAgents, int start, int end) {
	for (; start < end; start++)
	{
		Ped::Tagent* Agent = allAgents[start];
		Agent->computeNextDesiredPosition();
		Agent->setX(Agent->getDesiredX());
		Agent->setY(Agent->getDesiredY());
	}
}

void tickPTHREAD(std::vector<Ped::Tagent*> allAgents)
{	
	int numberOfThreads = 4;
	int groupsize = ceil(allAgents.size()/numberOfThreads);
	int size = allAgents.size() - (allAgents.size() % groupsize);
	for (int agentIndex = 0; agentIndex < size; agentIndex += groupsize)
	{
		std::thread t1(tickPTHREADAux, allAgents, agentIndex, agentIndex + groupsize);
		t1.join();
	}
	std::thread t2(tickPTHREADAux, allAgents, size, allAgents.size());
	t2.join();
}

void Ped::Model::tickOMP(std::vector<Ped::Tagent*> allAgents) {
	

	for (Tagent* Agent: allAgents)
	{
		Agent->computeNextDesiredPosition();
	}
	omp_set_num_threads(5);
	#pragma omp parallel sections
	{
		#pragma omp section 
		{
			updateHeatmap(implementation);
		}
		#pragma omp section 
		{
			move_agents_in_section(this->section1);
		}
		#pragma omp section 
		{	
			move_agents_in_section(this->section2);
		}
		#pragma omp section 
		{	
			move_agents_in_section(this->section3);
		}
		#pragma omp section 
		{
			move_agents_in_section(this->section4);
		}
	}
	move_agents_close_to_border();
}

void Ped::Model::tick() {
	std::vector<Tagent*> allAgents = getAgents();

	switch (implementation)
	{
	case Ped::CUDA:
		tickOMP(allAgents);
		break;
	case Ped::VECTOR:
		tickVECTOR(allAgents, this->arrayX, this->arrayY, this->destinationX, this->destinationY, this->destinationR);
		break;
	case Ped::OMP:
		tickOMP(allAgents);
		break;
	case Ped::PTHREAD:
		tickPTHREAD(allAgents);
		break;
	case Ped::SEQ:
		tickSEQ(allAgents);
		break;
	default:
		tickOMP(allAgents);
		break;
	}

}


/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

void Ped::Model::setup_section(section_t *section, int left_border, int right_border, section_t *section_to_left, section_t *section_to_right)
{
	section->left_border = left_border;
	section->right_border = right_border;
	section->section_to_left = section_to_left;
	section->section_to_right = section_to_right;
}

void Ped::Model::move_agents_in_section(section_t *section)
{
	for (Tagent *Agent : section->new_agents)
	{
		section->agents_in_section.push_back(Agent);
	}
	section->new_agents.clear();
	
	std::vector<Tagent *> agents_still_in_section;
	for (Tagent *Agent : section->agents_in_section)
	{
		if (is_agent_close_to_border(section, Agent))
		{
			#pragma omp critical
			{
				agents_to_move_SEQ.push_back(Agent);
			}
		}
		else
		{
			this->move(Agent);
			agents_still_in_section.push_back(Agent);
		}
	}
	section->agents_in_section = agents_still_in_section;
}

bool Ped::Model::is_agent_close_to_border(section_t *section, Tagent *Agent)
{
	if (section->left_border +2 > Agent->getX())
	{
		return true;
	}
	if (section->right_border -2 < Agent->getX())
	{
		return true;
	}
	return false;
}


void Ped::Model::agent_change_section(section_t *section, Tagent *Agent)
{
	if (section->left_border > Agent->getX())
	{
		if (section->section_to_left != NULL)
		{
			section->section_to_left->new_agents.push_back(Agent);
		}
	}
	else
	{
		if (section->section_to_right != NULL)
		{
			section->section_to_right->new_agents.push_back(Agent);
		}
	}
}

void Ped::Model::move_agents_close_to_border()
{
	for (Tagent *Agent : agents_to_move_SEQ)
	{
		this->move(Agent);

		int X = Agent->getX();
		if (X < 40)
		{
			this->section1->agents_in_section.push_back(Agent);
		}
		else if (X < 80)
		{
			this->section2->agents_in_section.push_back(Agent);
		}
		else if (X < 120)
		{
			this->section3->agents_in_section.push_back(Agent);
		}
		else if (X <= 160)
		{
			this->section4->agents_in_section.push_back(Agent);
		}
	}
	agents_to_move_SEQ.clear();
}


// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2,p3,p4;
	if (diffX == 0 || diffY == 0)
	{
		int diff = diffX + diffY;
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(agent->getX() + diff, agent->getY() + diff);
		p2 = std::make_pair(agent->getX() + diff, agent->getY() - diff);
		p3 = std::make_pair(agent->getX() - diff, agent->getY() + diff);
		p4 = std::make_pair(agent->getX() - diff, agent->getY() - diff);
	}
	else {
		// Agent wants to walk diagonally
		p1 = p3 = std::make_pair(pDesired.first, agent->getY());
		p2 = p4 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);
	prioritizedAlternatives.push_back(p3);
	prioritizedAlternatives.push_back(p4);
	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position	
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent) {delete agent; });
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination) {delete destination; });
}
