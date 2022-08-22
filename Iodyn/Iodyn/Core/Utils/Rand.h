#pragma once

#include <iostream>
#include <random>

std::default_random_engine generator;
namespace Idn
{

	int rndNi(int low, int high)
	{
		std::uniform_int_distribution<int> dist(low, high);

		return dist(generator);
	}

}