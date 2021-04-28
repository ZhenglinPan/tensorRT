#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
using namespace std;
// convert data(string type) received from JTX to int
// void dataConvert(int* rev_data, char *rev_string){
//     char *p; 
    
//     p = strtok(rev_string, ",");
    
//     int i = 0;
//     while (p){
//         rev_data[i++] = atoi(p);
//         p = strtok(NULL, ",");
//     }
// }
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

  for (int i = 0; i < 4; i++) {
//    Serial.print(rev_data[i]);
//    Serial.print(",");
  }
//  Serial.print("\r\n");/
}

int main(){
    char rev_string[] = "332, 104, 511, -1;";
    
    int rev_data[4];
    dataConvert(rev_data, rev_string);
    
    for(int i=0;i<4;i++){
        printf("%d\n", rev_data[i]);
    }

    return 0;
}

