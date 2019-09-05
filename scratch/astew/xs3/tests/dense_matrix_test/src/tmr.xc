

timer t;

unsigned start;

void timerStart()
{
    t:>start;
}

unsigned timerEnd()
{
    unsigned end;
    t:>end;
    return (end-start);
}