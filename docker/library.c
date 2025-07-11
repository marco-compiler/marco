int discreteLog (int base, int val)
/**
 *  input : base (>= 2), it is the logarithm base 
 *          val (> 0), it is the value which we want to calculate the logarithm of
*/ 
    {
        int cont = 0;

        if (base >= 2)
            {
                while (val > 1)
                    {
                        cont++;
                        val = val/base;
                    }  
            }
         

        return (cont);
    }