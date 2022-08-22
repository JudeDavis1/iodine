#pragma once

#include <iostream>


// Abstract class for all objects in a rendering window e.g: A sphere
class ObjectBase
{
public:
	ObjectBase() {}

	virtual void Begin() = 0;
	virtual void Render() = 0;
	virtual void End() = 0;

	virtual ~ObjectBase() = default;
};

