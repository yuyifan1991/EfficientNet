# EfficientNet
>**Install requirement**  
>>(1)pytorch version == v1.1  
>>(2)cuda version >= 10.0  
>>(3)python version >=3.6  

>**How to train your owe data**   
>>(1)Pepare your data, make the directory list as:
>>   * /your foldername  
>>       * --train  
>>         * --picture1  
>>         * --picture2  
>>         * -- ...  
>>       * --test  
>>         * --picture1  
>>         * --picture2  
>>         * -- ...  
>>      * --model  
  
>>(2) Run the ***train.py*** file  
>>>   $ python train.py  
  
>The weight model will be saved in /mode/xxx  
  
>**How to test picture in batch***  
>>Run the ***test.py*** file  
>>> $python test.py  
  
>**How to test picture in single*** 
>>Run the ***test-single.py*** file  
>>> $python test-single  


