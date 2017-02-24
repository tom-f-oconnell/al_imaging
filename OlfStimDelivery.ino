#include <RBD_Timer.h>

// if you comment out this define, the compiled code will exclude any serial prints
// to make things slightly more real time
#define SERIAL_OUTPUT

#define NOT_RANDOM 0
#define BLOCKS     1
#define TOTAL      2

const int randomize = TOTAL;
// NOT_RANDOM - don't randomize. will go from min to max_olfactometer pin, repeatNum times for each
// BLOCKS - randomize odor order, but keep presentations of each odor within blocks
// TOTAL - randomize completely. odor not kept in blocks.

const int scopePin = 3;

const int odor_signaling_pin = 12;   // will send a number of pulses = digital pin # pulsed for current trial

//set stimulus variables
const int parafin_pin = 4;
const int num_odors = 2;         // one of these should always be paraffin oil
const int num_repeats = 20;       // number of times each odor will be presented

const int ITI = 26;            // intertrial interval in seconds (30)
const int odorPulseLen = 2;    // length of the odor pulse in seconds
const int scopeLen = 4;       // length of scope acquisition time in seconds (30)
const int odorPulseOnset = 1; // onset time of odor pulse in seconds

// uses all pins in this range as digital outputs
// one valve per pin

// parafin is handled separately, and its valve is using pin 4
const int min_olfactometer_pin = 5;
const int max_olfactometer_pin = min_olfactometer_pin + num_odors - 1;
// should not write to this after setup
int odor_pins[num_odors * num_repeats];

int trialInd = 0;
int flag = 0;
int started = 0;

//create timer object
RBD::Timer trialTimer;
RBD::Timer OlfStartTimer;
RBD::Timer OlfLenTimer;

unsigned long t;

// the setup function runs once when you press reset or power the board
void setup() {
  pinMode(parafin_pin, OUTPUT);
  digitalWrite(parafin_pin, LOW);
  
  // so that I have time to start Thor software after pressing upload
  // once I get their source, the Arduino and Thor should play more nicely together
  delay(10000);
  
  #ifdef SERIAL_OUTPUT
    Serial.begin(9600);
  #endif

  // to draw odor order separately across runs of this code
  // must leave analog 0 floating for this to work!
  randomSeed(analogRead(0));

  // initialize digital pin as an output.
  // TODO remove or actually use
  pinMode(LED_BUILTIN, OUTPUT);   // built in LED, pulse when odor pulse is being delivered
  pinMode(scopePin, OUTPUT);
  pinMode(odor_signaling_pin, OUTPUT);

  for (int i = min_olfactometer_pin; i <= max_olfactometer_pin; i++) {
    pinMode(i, OUTPUT);
  }
  
  digitalWrite(scopePin, LOW);
  for (int i = min_olfactometer_pin; i <= max_olfactometer_pin; i++) {
    digitalWrite(i, LOW);
  }

#ifdef SERIAL_OUTPUT
  //checks serial
  Serial.println("Serial working");
#endif

  // TODO casting order correct? (not a problem now, but potential for overflow)
  trialTimer.setTimeout((unsigned long) scopeLen * 1000); // timer runs from start to end of a trial
  OlfStartTimer.setTimeout((unsigned long) odorPulseOnset * 1000);
  OlfLenTimer.setTimeout((unsigned long) odorPulseLen * 1000);

  int curr_odor;
  if (randomize == NOT_RANDOM) {

    curr_odor = min_olfactometer_pin;
    for (int i = 0; i < num_odors; i++) {
      for (int j = 0; j < num_repeats; j++) {
        odor_pins[i * num_repeats + j] = curr_odor;
      }
      curr_odor++;
    }

  } else if (randomize == BLOCKS) {

    boolean used[num_odors];

    for (int i = 0; i < num_odors; i++)
      used[i] = false;

    for (int i = 0; i < num_odors; i++) {

      // should nearly always terminate in reasonable time if num_odors > 0
      // because used[] is checked at end of each outer loop, and on first entry here is all false
      do {
        // picks uniformly from 0 to (num_odors - 1)
        curr_odor = random(num_odors);
      } while (used[curr_odor]);
      // will terminate as soon as it finds an odor that has not been used yet

      for (int j = 0; j < num_repeats; j++) {
        odor_pins[i * num_repeats + j] = curr_odor + min_olfactometer_pin;
      }

      used[curr_odor] = true;
    } // end for

  } else if (randomize == TOTAL) {

    // now used is an array of counters, counting the # of times an odor has been used
    // should be on [0,num_repeats]
    unsigned int used[num_odors];

    for (int i = 0; i < num_odors; i++) {
      used[i] = 0;
    }

    for (int i = 0; i < num_repeats * num_odors; i++) {
      do {
        curr_odor = random(num_odors);
      } while (used[curr_odor] >= num_repeats);

      odor_pins[i] = curr_odor + min_olfactometer_pin;
      used[curr_odor]++;
    }
  }

  #ifdef SERIAL_OUTPUT
    Serial.println("Planned stimulus order:");
    for (int i = 0; i < num_odors * num_repeats; i++) {
      Serial.print(odor_pins[i]);
      Serial.print(' ');
    }
    Serial.println("");
  #endif

  // TODO so she did try this?
  //trialTimer.restart();
}


// the loop function runs over and over again forever
void loop() {
  static int curr_odor;    // taken from the odor_pins array sequentially
  //trial begins

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
    curr_odor = odor_pins[trialInd];
    trialInd = trialInd + 1;

    OlfStartTimer.restart();
    digitalWrite(scopePin, HIGH);

    // it is important that this comes after turning the scope pin on
    // because this takes a few milliseconds
    signal_odor(curr_odor);

  #ifdef SERIAL_OUTPUT
    Serial.print("Trial # ");
    Serial.print(trialInd);
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
      digitalWrite(parafin_pin, HIGH);
      digitalWrite(curr_odor, HIGH);
      digitalWrite(odor_signaling_pin, HIGH);

  #ifdef SERIAL_OUTPUT
      Serial.print("\todor pulse start...");
      t = trialTimer.getValue();
      printTime(t);
  #endif
    }
    // TODO documentation makes it seem like it would work without flag (onExpired only returns true once per .restart())
    if (OlfLenTimer.onExpired() && flag == 1) {
      digitalWrite(parafin_pin, LOW);
      digitalWrite(curr_odor, LOW);
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

    if (trialInd == num_odors * num_repeats) {
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
}

#ifdef SERIAL_OUTPUT
void printTime(unsigned long t) {
  Serial.print("t = ");
  Serial.print(t);
  Serial.print(" ms");
  Serial.print("\n");
}
#endif



