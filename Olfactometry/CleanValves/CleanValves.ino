
const int min_pin = 5;
const int max_pin = 11;

void setup() {
  for (int i=min_pin;i<max_pin;i++) {
    pinMode(i, OUTPUT);
    digitalWrite(i, LOW);
  }
}

void loop() {
  for (int i=min_pin;i<max_pin;i++) {
    pinMode(i, OUTPUT);
    digitalWrite(i, HIGH);
  }
  delay(20000);
  for (int i=min_pin;i<max_pin;i++) {
    pinMode(i, OUTPUT);
    digitalWrite(i, LOW);
  }
  delay(1000);
}
