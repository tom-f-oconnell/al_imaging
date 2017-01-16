#include <RBD_Timer.h>



//set stimulus variables
const int trialNum = 3;      //number of trials
const int ITI = 30;            //intertrial interval in seconds
const int odorPulseLen = 2;  //length of the odor pulse in seconds
const int scopeLen = 30;       //length of scope acquisition time in seconds
const int odorPulseOnset = 10; //onset time of odor pulse in seconds


//pin assigments

int scopeDispPin = 3; 
int olfactometerPin = 5; //6 is 2-butanone, 5 is pentyl acetate
int scopePin=13;

int trialInd = 0;

int flag = 0;
int started=0;

//create timer object
RBD::Timer trialTimer;
RBD::Timer OlfStartTimer;
RBD::Timer OlfLenTimer;

unsigned long t;


// the setup function runs once when you press reset or power the board
void setup() {
  //initialize serial
  Serial.begin(9600);

  // initialize digital pin as an output.
  pinMode(LED_BUILTIN, OUTPUT);   // built in LED, pulse when odor pulse is being delivered
  pinMode(scopeDispPin, OUTPUT);
  pinMode(scopePin, OUTPUT);
  pinMode(olfactometerPin, OUTPUT);

  digitalWrite(scopeDispPin, LOW);
  digitalWrite(scopePin, LOW);
  digitalWrite(olfactometerPin, LOW);
  //checks serial
  Serial.println("Serial working");

  trialTimer.setTimeout((unsigned long) scopeLen * 1000); //timer runs from start to end of a trial
  OlfStartTimer.setTimeout((unsigned long) odorPulseOnset * 1000);
  OlfLenTimer.setTimeout((unsigned long) odorPulseLen * 1000);
  

  //trialTimer.restart();


}


// the loop function runs over and over again forever
void loop() {

  //trial begins

  // TODO why not just put this in setup? it seems equivalent
  if (started==0){

    delay(2000);

    Serial.println("------------Session start------------");
    trialTimer.restart();
    started=started+1;
  }
  
  if (trialTimer.onActive()) {
    // TODO remove. not used until redefind.
    t = trialTimer.getValue();
    trialInd = trialInd + 1;
    OlfStartTimer.restart();
    digitalWrite(scopeDispPin, HIGH);
    digitalWrite(scopePin, HIGH);
    Serial.print("Trial # ");
    Serial.print(trialInd);
    Serial.print("/");
    Serial.print(trialNum);
    Serial.print("................");
    t = trialTimer.getValue();
    printTime(t);
  } // end onActive check

  if (trialTimer.isActive()) {
    if (OlfStartTimer.onExpired()) {
      OlfLenTimer.restart();
      flag = 1;
    }
    if (OlfLenTimer.onActive()) {
      digitalWrite(olfactometerPin, HIGH);
      Serial.print("\todor pulse start...");
      t = trialTimer.getValue();
      printTime(t);

    }
    if (OlfLenTimer.onExpired() && flag == 1) {
      digitalWrite(olfactometerPin, LOW);
      Serial.print("\todor pulse stop....");
      t = trialTimer.getValue();
      printTime(t);

    }
  } // end isActive check

  if (trialTimer.onExpired()) {
    digitalWrite(scopeDispPin, LOW);
    digitalWrite(scopePin, LOW);
    t = trialTimer.getValue();
    Serial.println("Trial complete...........");
    printTime(t);

    if (trialInd == trialNum) {
      trialTimer.stop();
      Serial.println("------------Session complete------------");
      digitalWrite(olfactometerPin, LOW);
      digitalWrite(scopeDispPin, LOW);
      digitalWrite(scopePin, LOW);
    }
    else {
      delay(ITI * 1000);
      trialTimer.restart();
    }
  } // end onExpired check
  
} // end loop

void printTime(unsigned long t) {
  Serial.print("t = "); Serial.print(t); Serial.print(" ms");
  Serial.print("\n");
}



