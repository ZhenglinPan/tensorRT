#include<string>

void setup()
{
  Serial.begin(9600);//设置波特率
}

void loop()
{
  string a;
  a = Serial.read();//读取串口内容
  if (Serial.available())
  {
    Serial.print(a);
//    if (a == '1')
//    {
//      Serial.print("hello!");
//    }
  }
}
