#include<stdio.h>
#include<stdlib.h>
#include<string.h>

String comdata = "";

void dataConvert(int* rev_data, char *rev_string){
    char *p; 
    
    p = strtok(rev_string, ",");
    
    int i = 0;
    while (p){
        rev_data[i++] = atoi(p);
        p = strtok(NULL, ",");
    }
}

void setup() 
{
  Serial.begin(9600);      //设定的波特率
}

void loop() 
{
   comdata = "";
   while (Serial.available() > 0){
        comdata += char(Serial.read());
        delay(2);
    }
    
   if (comdata.length() > 0){
       Serial.println(comdata);
    }
    
    char rev_string[20];
    strcpy(rev_string, comdata.c_str());
    
    int rev_data[4];
    dataConvert(rev_data, rev_string);
    
}
