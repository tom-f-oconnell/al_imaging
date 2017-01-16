#include <RBD_Timer.h>

// if you comment out this define, the compiled code will exclude any serial prints
// to make things slightly more real time
#define SERIAL_OUTPUT

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
  #ifdef SERIAL_OUTPUT
  Serial.begin(9600);
  #endif

  // initialize digital pin as an output.
  // TODO remove or actually use
  pinMode(LED_BUILTIN, OUTPUT);   // built in LED, pulse when odor pulse is being delivered
  pinMode(scopeDispPin, OUTPUT);
  pinMode(scopePin, OUTPUT);
  pinMode(olfactometerPin, OUTPUT);

  digitalWrite(scopeDispPin, LOW);
  digitalWrite(scopePin, LOW);
  digitalWrite(olfactometerPin, LOW);

  #ifdef SERIAL_OUTPUT
  //checks serial
  Serial.println("Serial working");
  #endif

  trialTimer.setTimeout((unsigned long) scopeLen * 1000); //timer runs from start to end of a trial
  OlfStartTimer.setTimeout((unsigned long) odorPulseOnset * 1000);
  OlfLenTimer.setTimeout((unsigned long) odorPulseLen * 1000);

  // TODO so she did try this?
  //trialTimer.restart();
}


// the loop function runs over and over again forever
void loop() {

  //trial begins

  // TODO why not just put this in setup? it seems equivalent
  if (started==0){

    delay(2000);

    #ifdef SERIAL_OUTPUT
    Serial.println("------------Session start------------");
    #endif
    
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

    #ifdef SERIAL_OUTPUT
    Serial.print("Trial # ");
    Serial.print(trialInd);
    Serial.print("/");
    Serial.print(trialNum);
    Serial.print("................");
    t = trialTimer.getValue();
    printTime(t);
    #endif
  } // end onActive check

  if (trialTimer.isActive()) {
    if (OlfStartTimer.onExpired()) {
      OlfLenTimer.restart();
      flag = 1;
    }
    if (OlfLenTimer.onActive()) {
      digitalWrite(olfactometerPin, HIGH);
      
      #ifdef SERIAL_OUTPUT
      Serial.print("\todor pulse start...");
      t = trialTimer.getValue();
      printTime(t);
      #endif
    }
    if (OlfLenTimer.onExpired() && flag == 1) {
      digitalWrite(olfactometerPin, LOW);

      #ifdef SERIAL_OUTPUT
      Serial.print("\todor pulse stop....");
      t = trialTimer.getValue();
      printTime(t);
      #endif
    }
  } // end isActive check

  if (trialTimer.onExpired()) {
    digitalWrite(scopeDispPin, LOW);
    digitalWrite(scopePin, LOW);

    #ifdef SERIAL_OUTPUT
    t = trialTimer.getValue();
    Serial.println("Trial complete...........");
    printTime(t);
    #endif

    if (trialInd == trialNum) {
      trialTimer.stop();

      #ifdef SERIAL_OUTPUT
      Serial.println("------------Session complete------------");
      #endif
      
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

#ifdef SERIAL_OUTPUT
void printTime(unsigned long t) {
  Serial.print("t = ");
  Serial.print(t);
  Serial.print(" ms");
  Serial.print("\n");
}
#endif



