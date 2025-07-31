double discreteLog (int base, int val, int aa)
    {
        double res = 0.0; 
        while (val > 1)
            {
                res = res + 1; 
                val = val/base;
            }   
        return res; 
    }