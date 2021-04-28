#include<stdio.h>
#include<stdlib.h>
#include<string.h>

String comdata;
int rev_data[4];

void dataConvert(int* rev_data, char *rev_string) {
  char *p;
  int i;
  for (i = 0; rev_string[i] != ';' && i < strlen(rev_string); i++);

  rev_string[i] = '\0';

  p = strtok(rev_string, ",");

  i = 0;
  while (p) {
    rev_data[i++] = atoi(p);
    p = strtok(NULL, ",");
  }
  char buff[50];
  for (int i = 0; i < 4; i++) {
    Serial.print(rev_data[i]);
    Serial.print(",");
  }
}

void setup()
{
  Serial.begin(115200);      //设定的波特率
}

void loop()
{

  while (Serial.available() > 0) {
    char ch = char(Serial.read());
    if (ch == ';') {
      while(Serial.available() > 0) Serial.read();
      dataConvert(rev_data, comdata.c_str());
      comdata = "";
    }
    else
    {
      comdata += ch;
    }
   
  }

  

}
