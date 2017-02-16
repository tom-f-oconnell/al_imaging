
#define LED 13

const int num_odors = 8;
const int min_olfactometer_pin = 4;
const int max_olfactometer_pin = min_olfactometer_pin + num_odors - 1;

void setup() {
  Serial.begin(9600);
  Serial.print('Testing olfactometer valves on Arduino pins ');
  Serial.print(min_olfactometer_pin);
  Serial.print(' through ');
  Serial.println(max_olfactometer_pin);
  
  for (int i=min_olfactometer_pin;i<=max_olfactometer_pin;i++) {
    pinMode(i, OUTPUT);
  }
  
  for (int i=min_olfactometer_pin;i<=max_olfactometer_pin;i++) {
    digitalWrite(i, LOW);
  }
}

void loop() {
  for (int i=min_olfactometer_pin;i<=max_olfactometer_pin;i++) {
    Serial.println(i);
    digitalWrite(i, HIGH);
    digitalWrite(LED, HIGH);
    delay(1000);
    digitalWrite(i, LOW);
    digitalWrite(LED, LOW);
    delay(1000);
  }
}
