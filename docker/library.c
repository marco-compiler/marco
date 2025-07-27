double discreteLog (int base, int val)
    {
        double cont = 0.0;
        while (val > 1)
            {
                cont++;
                val = val/base;
            }   

        return (cont);
    }