# TemperatureControlledUnit
Design a temperature controller device that will continuously sense two reference temperatures, and control working of the motor based on that. 
link for Doppler Sensor :- https://github.com/aethernavshulkraven-allain/Ultrasonic-Doppler-Sensor-for-Motion-Detection

# Statement:
Design a temperature controller device that will continuously sense two reference temperatures, which are 30OC (Lower Limit: TL) and 35OC (Higher Limit: TH), such that if the temperature goes below TL, the unit will activate Motor 1, and hence if the temperature goes above TH, Motor 2 will be activated. If the temperature of the sensing element is maintained between the given TH and TL, both the Motors should be switched OFF. here an amplifier is introduced to magnify the change because of temperature variations. The amplifier shall operate in the linear region between the given temperature ranges.

# Technical Specifications:
The product should meet the following technical specifications:
a. This product should be able to work with the LM35 temperature sensor. The designer can use additional circuits for increased sensitivity.
b. The motor will be a DC motor with a 300 RPM rating and normally operates on 12 V DC.
c. The higher and lower reference voltages(VR1 and VR2) need to be decided by the designer
suitably, to suit the given temperature band requirements.
d. Only two power supplies should be provided externally, and all the supplies need to be generated in the circuit using voltage dividers.


# Restrictions
a. The amplification part should be implemented using BJTs/MOSFETs; the use of op-amps is NOT allowed for this purpose.
b. Students can use active and passive filters per their requirements, and use IC 741 to implement comparison logic block.
![image](https://github.com/aethernavshulkraven-allain/TemperatureControlledUnit/assets/112583472/d3860a11-98e9-4855-a296-5238f79beb10)


# Our Approach

## BJT vs MOSFET
We opted for BJT 2n2222 IC for Amplification, 
we chose BJT as,

1. Voltage and Current Requirements: If your application requires amplification of signals with relatively low voltage and current levels, a BJT might be more suitable as it typically has lower voltage and current requirements compared to MOSFETs.

2. Temperature Sensitivity: BJTs generally exhibit less sensitivity to temperature variations compared to MOSFETs. In a temperature-controlled unit, where maintaining stability is crucial, this characteristic might make BJTs a preferred choice.

3. Cost Considerations: BJTs are often less expensive than MOSFETs, making them a more economical choice, especially if your project has budget constraints.

4. Amplification Linearity: Depending on the specific requirements of your amplifier circuit, you might find that BJTs offer better linearity over a wider range of signal amplitudes compared to MOSFETs.

5. Ease of Implementation: BJTs are typically easier to implement in certain amplifier configurations, such as the cascode amplifier, due to their simpler biasing and operating characteristics.

Overall, the choice between BJT and MOSFET for project would depended on a careful consideration of these factors along with the specific requirements and constraints of temperature-controlled unit.

## Amplification
As per our problem statement, LM35 was to be used as our sensor, and due to its small output, we had to use an Amplifier. We were not allowed to use diode for amplification and had to be done via BJT or MOSFET.

while exploring various amplification models and circuits we found a Cascode Configuration as best as,


1. Improved Gain and Bandwidth: The cascode configuration provides higher gain and wider bandwidth compared to other common amplifier configurations. This can be beneficial for accurately amplifying temperature-related signals in your unit.

2. Enhanced Linearity: Cascode amplifiers typically exhibit better linearity, which is crucial for maintaining the accuracy of temperature measurements and control in your unit.

3. Reduced Miller Effect: The cascode configuration helps reduce the Miller effect, which can degrade the amplifier's high-frequency performance. This is important for ensuring stable operation across a range of temperatures.

4. Improved Stability: Cascode amplifiers offer improved stability, which is essential in a temperature-controlled system where fluctuations in temperature can impact the performance of the amplifier.

5. Lower Output Impedance: The cascode configuration typically results in lower output impedance, which can improve the overall efficiency and performance of your temperature-controlled unit.

6. Minimized Noise: Cascode amplifiers can help minimize noise, which is critical for accurate temperature sensing and control applications where even small fluctuations can affect the system's performance.

Overall, the cascode configuration offers a robust solution for amplifying temperature-related signals in unit, providing high gain, improved linearity, stability, and noise performance, making it a suitable choice for project.

### Usage of capacitors
Capacitors are commonly used in amplifiers for various reasons:

Coupling Capacitors: Amplifiers often consist of multiple stages, such as voltage amplification stages followed by a power amplification stage. Coupling capacitors are used to block DC voltage while allowing AC signals to pass between stages. This prevents DC bias from one stage affecting the next stage, while still allowing the AC signal to be amplified.

Filtering Capacitors: Capacitors can be used in conjunction with resistors to form RC filter circuits, which filter out unwanted frequencies from the amplified signal. High-pass, low-pass, band-pass, and band-stop filters can be created using capacitors in amplifiers to tailor the frequency response according to the application's requirements.

Biasing and Stability: Capacitors can be used for biasing circuits to stabilize the amplifier's operating point or to provide AC feedback for stability. By coupling AC signals while blocking DC, capacitors help maintain the amplifier's desired operating conditions.

Signal Integrity: Capacitors can help improve the signal integrity by blocking any DC offsets or unwanted bias voltages from reaching the output, ensuring that only the desired AC signal is amplified.

Decoupling Capacitors: Decoupling capacitors are used to provide a stable voltage supply to amplifier stages by filtering out any high-frequency noise or voltage fluctuations from the power supply. This ensures that the amplifier operates with a clean and stable DC voltage, minimizing distortion and improving performance.

Overall, capacitors play a crucial role in amplifier circuits by enabling signal coupling, filtering, biasing, stability, and voltage regulation, contributing to the amplifier's overall functionality and performance.

![Unknown-1](https://github.com/aethernavshulkraven-allain/TemperatureControlledUnit/assets/112583472/5cf924f5-42c1-4ad2-bb68-ad6e0605ffa1)

Our gain : _26.67 - 30 V/V_

## Understanding Miller Effect
_source : Behzad Razavi YouTube Lectures_

The Miller effect is a phenomenon observed in amplifier circuits, particularly in configurations with capacitive coupling between the input and output stages. It occurs when the voltage across the input and output capacitances of a transistor amplifier creates an effective feedback path, leading to a reduction in the amplifier's bandwidth and potentially introducing instability.

In a standard common-emitter amplifier configuration, the input capacitance (Cπ) of the transistor, which is typically small, couples with the output capacitance (Cμ) between the collector and base. This coupling effectively increases the input capacitance seen by the input signal, resulting in a lower effective bandwidth and degraded high-frequency performance.

The cascode configuration mitigates the Miller effect by decoupling the input and output stages of the amplifier using a second transistor. In a cascode amplifier, the input signal is applied to the base of the first transistor (the common-emitter stage), while the second transistor (the common-base stage) is connected in series with the first transistor's collector. This arrangement effectively isolates the input capacitance of the first transistor from the output capacitance, preventing the capacitances from interacting and reducing the Miller effect.

Since the input capacitance of the first transistor is no longer directly coupled to the output capacitance, the effective input capacitance seen by the input signal is significantly reduced. As a result, the cascode configuration allows for higher bandwidth and improved high-frequency performance compared to traditional amplifier configurations.

In summary, the cascode configuration minimizes the Miller effect by decoupling the input and output stages of the amplifier, effectively isolating the input capacitance from the output capacitance and preserving the amplifier's bandwidth and stability, which is crucial for applications like temperature-controlled units.

https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FInverting-amplifier-with-a-Miller-capacitance-and-b-equivalent-model_fig6_323465497&psig=AOvVaw0emjAlylNmZ3rwMcE6IPke&ust=1718898027972000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCPCT9KiA6IYDFQAAAAAdAAAAABAI![image](https://github.com/aethernavshulkraven-allain/TemperatureControlledUnit/assets/112583472/9a11d5b4-2b90-4e18-aa79-5f2381cfe82a)

## Comparator Logic

## Issues Faced
When I connected my amplifier with an window comparator circuit I was facing issue of loading what is it why was it occurring,

to our understanding, 
its likely due to impedance mismatch between the output of the amplifier and the input of the comparator. This impedance mismatch can cause a distortion of the signal and affect the performance of both the amplifier and the comparator.

### Why it Occurs:

a. Impedance Mismatch: The output impedance of your amplifier might not match the input impedance of the window comparator circuit. This can result in a mismatched transfer of power between the two circuits, causing loading effects.
b. Signal Distortion: The loading effect can distort the output signal of the amplifier, affecting its accuracy and stability. It can also impact the threshold levels of the window comparator, leading to incorrect comparisons.

![Unknown-3](https://github.com/aethernavshulkraven-allain/TemperatureControlledUnit/assets/112583472/3565d1b1-76bc-4060-865a-0fbdd24a5dbb)

![Unknown-2](https://github.com/aethernavshulkraven-allain/TemperatureControlledUnit/assets/112583472/404fca63-c92f-4599-babf-953d5585ab50)
