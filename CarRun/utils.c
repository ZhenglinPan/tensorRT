#include<stdio.h>
#include<stdlib.h>
#include<string.h>

// convert data(string type) received from JTX to int
void dataConvert(int* rev_data, char *rev_string){
    char *p; 
    
    p = strtok(rev_string, ",");
    
    int i = 0;
    while (p){
        rev_data[i++] = atoi(p);
        p = strtok(NULL, ",");
    }
}


int main(){
    char rev_string[] = "46, 402, 312, 1";
    
    int rev_data[4];
    dataConvert(rev_data, rev_string);
    
    for(int i=0;i<4;i++){
        printf("%d\n", rev_data[i]);
    }

    return 0;
}

