#include "Application.h"
#include <Iodine/Core/Engine/Engine.h>


void TestEngine();

int main() {
	// Application application("Iodine", 500, 500);
	// application.Run();
	TestEngine();
}

void TestEngine() {
	Idn::Engine engine;
	engine.Run();
}

