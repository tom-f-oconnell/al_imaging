#include <RBD_Timer.h>

// if you comment out this define, the compiled code will exclude any serial prints
// to make things slightly more real time
#define SERIAL_OUTPUT

const int scopePin = 3;
const int odor_signaling_pin = 12;   // will send a number of pulses = digital pin # pulsed for current trial

//set stimulus variables
const int balance_pin = 4;
const int pentyl_acetate_pin = 5;
const int num_odors = 6;         // one of these should always be paraffin oil
const int num_repeats = 5;       // number of times each odor will be presented

const int ITI = 30;            // intertrial interval in seconds
const int odorPulseLen_ms = 500;    // length of the odor pulse in milliseconds
const int scopeLen = 15;       // length of scope acquisition time in seconds (30) (was using 10)
const int odorPulseOnset = 3; // onset time of odor pulse in seconds (3) (maybe increase)

// uses all pins in this range as digital outputs
// one valve per pin

// parafin is handled separately, and its valve is using pin 4
const int min_olfactometer_pin = 6;
const int max_olfactometer_pin = min_olfactometer_pin + num_odors - 1;

// will hold an integer corresponding to pin number to be pulsed in each index
// -1 in other indices
int odor_buffer[num_odors];

int trial_index = 0;
int flag = 0;
int started = 0;

//create timer object
RBD::Timer trialTimer;
RBD::Timer OlfStartTimer;
RBD::Timer OlfLenTimer;

unsigned long t;

// signal to the data acquisition which olfactometer pin we will pulse for this trial
// ~2 ms period square wave. # pulses = pin #
void signal_odor(unsigned int pin) {
  digitalWrite(odor_signaling_pin, LOW);
  delay(1);
  while (pin > 0) {
    digitalWrite(odor_signaling_pin, HIGH);
    delay(1);
    digitalWrite(odor_signaling_pin, LOW);
    delay(1);
    pin--;
  }
  delay(10);
}

#ifdef SERIAL_OUTPUT
void printTime(unsigned long t) {
  Serial.print("t = ");
  Serial.print(t);
  Serial.print(" ms");
  Serial.print("\n");
}
#endif

/** Sends the trial_index to the listening Python script.
 *  Expects a list of integers, one per line, 
 *  that are the pins to be used on the current trial.
 *  Expects a -1 after list of pins.
 *  Enters these integers in to the odor_buffer.
 */
void fill_odor_buffer() {
  Serial.println(trial_index);

  for (int i=0;i<num_odors;i++) {
    // TODO test. need to cast?
    odor_buffer[i] = Serial.parseInt();
    if (odor_buffer[i] == -1)
      break;
  }
}

void output_odor_buffer() {
  Serial.println("odor_buffer:");
  for (int i=0;i<num_odors;i++) {
     Serial.println(odor_buffer[i]);
  }
}

// the setup function runs once when you press reset or power the board
void setup() {
  pinMode(balance_pin, OUTPUT);
  pinMode(pentyl_acetate_pin, OUTPUT);
  digitalWrite(balance_pin, LOW);
  digitalWrite(pentyl_acetate_pin, LOW);
  
  // initialize digital pin as an output.
  // TODO remove or actually use
  pinMode(LED_BUILTIN, OUTPUT);   // built in LED, pulse when odor pulse is being delivered
  digitalWrite(LED_BUILTIN, LOW);
  
  pinMode(scopePin, OUTPUT);
  digitalWrite(scopePin, LOW);
  
  pinMode(odor_signaling_pin, OUTPUT);
  digitalWrite(odor_signaling_pin, LOW);

  // TODO setting one too many pins?
  for (int i = min_olfactometer_pin; i <= max_olfactometer_pin; i++) {
    pinMode(i, OUTPUT);
  }
  for (int i = min_olfactometer_pin; i <= max_olfactometer_pin; i++) {
    digitalWrite(i, LOW);
  }
  // indices not holding a pin to be used in a trial should be -1
  for (int i=0;i<num_odors;i++) {
    odor_buffer[i] = -1;
  }
  
  // so that I have time to start Thor software after pressing upload
  // once I get their source, the Arduino and Thor should play more nicely together
  delay(10000);
  
  #ifdef SERIAL_OUTPUT
  Serial.begin(9600);
  #endif

  // to draw odor order separately across runs of this code
  // must leave analog 0 floating for this to work!
  randomSeed(analogRead(0));

  #ifdef SERIAL_OUTPUT
  //checks serial
  Serial.println("Serial working");
  #endif

  // TODO casting order correct? (not a problem now, but potential for overflow)
  trialTimer.setTimeout((unsigned long) scopeLen * 1000); // timer runs from start to end of a trial
  OlfStartTimer.setTimeout((unsigned long) odorPulseOnset * 1000);
  OlfLenTimer.setTimeout((unsigned long) odorPulseLen_ms);

  // TODO so she did try this?
  //trialTimer.restart();
}


// the loop function runs over and over again forever
void loop() {
  // TODO why not just put this in setup? it seems equivalent
  if (started == 0) {

    delay(2000);

    #ifdef SERIAL_OUTPUT
    Serial.println("------------Session start------------");
    #endif

    trialTimer.restart();
    started = started + 1;
  }

  if (trialTimer.onActive()) {
    trial_index = trial_index + 1;

    OlfStartTimer.restart();
    digitalWrite(scopePin, HIGH);

    // this will modify the buffer accessed below
    fill_odor_buffer();

    // it is important that this comes after turning the scope pin on
    // because this takes ~20 milliseconds per odor (though could be made faster)
    for (int i=0;i<num_odors;i++) {
      if (odor_buffer[i] != -1)
        signal_odor(odor_buffer[i]);
    }

    #ifdef SERIAL_OUTPUT
    Serial.print("Trial # ");
    Serial.print(trial_index);
    Serial.print("/");
    Serial.print(num_odors * num_repeats);
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
      digitalWrite(balance_pin, HIGH);
      for (int i=0;i<num_odors;i++) {
        if (odor_buffer[i] != -1)
          digitalWrite(odor_buffer[i], HIGH);
      }
      digitalWrite(odor_signaling_pin, HIGH);

      #ifdef SERIAL_OUTPUT
      Serial.print("\todor pulse start...");
      t = trialTimer.getValue();
      printTime(t);
      #endif
    }
    // TODO documentation makes it seem like it would work without flag (onExpired only returns true once per .restart())
    if (OlfLenTimer.onExpired() && flag == 1) {
      digitalWrite(balance_pin, LOW);
      for (int i=0;i<num_odors;i++) {
        if (odor_buffer[i] != -1) {
          digitalWrite(odor_buffer[i], LOW);
          odor_buffer[i] = -1;
        }
      }
      digitalWrite(odor_signaling_pin, LOW);

      #ifdef SERIAL_OUTPUT
      Serial.print("\todor pulse stop....");
      t = trialTimer.getValue();
      printTime(t);
      #endif
    }
  } // end isActive check

  if (trialTimer.onExpired()) {
    digitalWrite(scopePin, LOW);

    #ifdef SERIAL_OUTPUT
    t = trialTimer.getValue();
    Serial.println("Trial complete...........");
    printTime(t);
    #endif

    if (trial_index == num_odors * num_repeats) {
      trialTimer.stop();

      #ifdef SERIAL_OUTPUT
      Serial.println("------------Session complete------------");
      #endif

      for (int i = min_olfactometer_pin; i <= max_olfactometer_pin; i++) {
        digitalWrite(i, LOW);
      }
      digitalWrite(scopePin, LOW);
    } else {
      // TODO be careful mixing delays and timers
      // CHECK!!!
      delay(ITI * 1000);
      trialTimer.restart();
    }
  } // end onExpired check
} // end loop

