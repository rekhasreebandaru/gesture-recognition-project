// Gesture Recognition with Virtual IoT Control - Arduino relay receiver.
// Commands over USB serial: LIGHT_ON, LIGHT_OFF, FAN_ON, FAN_OFF.

const unsigned long SERIAL_BAUD_RATE = 9600;
const int LIGHT_RELAY_PIN = 7;
const int FAN_RELAY_PIN = 8;

// Set this after checking the relay module datasheet/wiring.
const bool RELAY_ACTIVE_HIGH = true;

void setRelay(int relayPin, bool enabled) {
  bool outputLevel = RELAY_ACTIVE_HIGH ? enabled : !enabled;
  digitalWrite(relayPin, outputLevel ? HIGH : LOW);
}

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  pinMode(LIGHT_RELAY_PIN, OUTPUT);
  pinMode(FAN_RELAY_PIN, OUTPUT);
  setRelay(LIGHT_RELAY_PIN, false);
  setRelay(FAN_RELAY_PIN, false);
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();

  if (command == "LIGHT_ON") {
    setRelay(LIGHT_RELAY_PIN, true);
  } else if (command == "LIGHT_OFF") {
    setRelay(LIGHT_RELAY_PIN, false);
  } else if (command == "FAN_ON") {
    setRelay(FAN_RELAY_PIN, true);
  } else if (command == "FAN_OFF") {
    setRelay(FAN_RELAY_PIN, false);
  }
}
