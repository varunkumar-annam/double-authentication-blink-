class Simple{
public class main{
int met(int k){
int j;
if(k>1){

for(j=2;j<=k;j++){
if(k%j==0){
return 1;
}
}
return 0;
}
}
}
public static void main(String args[]){
int i;
for(i=0;i<=10;i++){
if(i%2==1){
System.out.println(i);
}
}
}
}