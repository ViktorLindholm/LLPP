///////////////////////////////////////////////////
// Low Level Parallel Programming 2017.
//
// 
//
// The main starting point for the crowd simulation.
//



#undef max
#include <Windows.h>
#include "ped_model.h"
#include "MainWindow.h"
#include "ParseScenario.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QApplication>
#include <QTimer>
#include <thread>

#include "PedSimulation.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstring>

#pragma comment(lib, "libpedsim.lib")

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

int main(int argc, char*argv[]) {
	bool timing_mode = 0;
	int i = 1;
	enum Ped::IMPLEMENTATION mode;
	enum Ped::IMPLEMENTATION timeMode;
	cout << "Choose testing: \n 1. Timing version \n 2. Graphics version\n";
	int testing = getchar();
	if (testing=='1')
	{
		timing_mode = 1;
		cout << "Compare SEQ to...\n 1. OMP \n 2. PThread \n 3. VECTOR \n";
		getchar();
		int timeChoice = getchar();
		switch (timeChoice)
		{
		case ('1'):
			timeMode = Ped::OMP;
			break;
		case ('2'):
			timeMode = Ped::PTHREAD;
			break;
		case ('3'):
			timeMode = Ped::VECTOR;
			break;
		default:
			timeMode = Ped::SEQ;
			break;
		}
	}
	else
	{
		cout << "Choose an implementation: \n 1. SEQ \n 2. OMP \n 3. PTHREAD \n 4. VECTOR\n\n";
		getchar();
		int choice = getchar();
		switch (choice)
		{
		case ('1'):
			mode = Ped::SEQ;
			break;
		case ('2'):
			mode = Ped::OMP;
			break;
		case ('3'):
			mode = Ped::PTHREAD;
			break;
		case ('4'):
			mode = Ped::VECTOR;
			break;
		default:
			mode = Ped::SEQ;
			break;
		}
	}
	QString scenefile = "hugeScenario.xml";
	

	// Enable memory leak check. This is ignored when compiling in Release mode. 
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	// Argument handling
	while (i < argc)
	{
		if (argv[i][0] == '-' && argv[i][1] == '-')
		{
			if (strcmp(&argv[i][2], "timing-mode") == 0)
			{
				cout << "Timing mode on\n";
				timing_mode = true;
			}
			else if (strcmp(&argv[i][2], "help") == 0)
			{
				cout << "Usage: " << argv[0] << " [--help] [--timing-mode] [scenario]" << endl;
				return 0;
			}
			else
			{
				cerr << "Unrecognized command: \"" << argv[i] << "\". Ignoring ..." << endl;
			}
		}
		else // Assume it is a path to scenefile
		{
			scenefile = argv[i];
		}

		i += 1;
	}
	int retval = 0;
	{ // This scope is for the purpose of removing false memory leak positives

		// Reading the scenario file and setting up the crowd simulation model
		Ped::Model model;
		ParseScenario parser(scenefile);
		model.setup(parser.getAgents(), parser.getWaypoints(), mode);

		// GUI related set ups
		QApplication app(argc, argv);
		MainWindow mainwindow(model);

		// Default number of steps to simulate. Feel free to change this.
		const int maxNumberOfStepsToSimulate = 100;
		
				

		// Timing version
		// Run twice, without the gui, to compare the runtimes.
		// Compile with timing-release to enable this automatically.
		if (timing_mode)
		{
			// Run sequentially

			double fps_seq, fps_target;
			{
				Ped::Model model;
				ParseScenario parser(scenefile);
				model.setup(parser.getAgents(), parser.getWaypoints(), Ped::SEQ);
				PedSimulation simulation(model, mainwindow);
				// Simulation mode to use when profiling (without any GUI)
				std::cout << "Running reference version...\n";
				auto start = std::chrono::steady_clock::now();
				simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
				auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
				fps_seq = ((float)simulation.getTickCount()) / ((float)duration_seq.count())*1000.0;
				cout << "Reference time: " << duration_seq.count() << " milliseconds, " << fps_seq << " Frames Per Second." << std::endl;
			}

			// Change this variable when testing different versions of your code. 
			// May need modification or extension in later assignments depending on your implementations
			{
				Ped::Model model;
				ParseScenario parser(scenefile);
				model.setup(parser.getAgents(), parser.getWaypoints(), timeMode);
				PedSimulation simulation(model, mainwindow);
				// Simulation mode to use when profiling (without any GUI)
				std::cout << "Running target version...\n";
				auto start = std::chrono::steady_clock::now();
				simulation.runSimulationWithoutQt(maxNumberOfStepsToSimulate);
				auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
				fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
				cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
			}
			std::cout << "\n\nSpeedup: " << fps_target / fps_seq << std::endl;
			
			

		}
		// Graphics version
		else
		{

			PedSimulation simulation(model, mainwindow);

			cout << "Demo setup complete, running ..." << endl;

			// Simulation mode to use when visualizing
			auto start = std::chrono::steady_clock::now();
			mainwindow.show();
			simulation.runSimulationWithQt(maxNumberOfStepsToSimulate);
			retval = app.exec();

			auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
			float fps = ((float)simulation.getTickCount()) / ((float)duration.count())*1000.0;
			cout << "Time: " << duration.count() << " milliseconds, " << fps << " Frames Per Second." << std::endl;
			
		}

		

		
	}
	_CrtDumpMemoryLeaks();

	cout << "Done" << endl;
	cout << "Type Enter to quit.." << endl;
	getchar(); // Wait for any key. Windows convenience...
	getchar();
	return retval;
}
