void discreteLog (int base, int val, int * res)
    {
        * res = 0.0; 
        while (val > 1)
            {
                *res = *res + 1; 
                val = val/base;
            }   
    }