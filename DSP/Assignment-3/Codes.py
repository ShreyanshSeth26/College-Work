from numpy import abs, log10, pi, cos, array, ceil, multiply
import matplotlib.pyplot as plt
from scipy.signal import kaiserord, firwin, freqz, lfilter, remez, freqz

question = 2
while question in [2, 3]:
    question = int(input("Please enter a Question 2 or 3: "))
    print()
    
    if  question == 2: 
        passband_edge, stopband_edge = pi/8,pi/4
        passband_ripple, stopband_ripple = 40, 50
        M, beta = kaiserord(max(passband_ripple, stopband_ripple), ((stopband_edge - passband_edge) / pi))
        filter_coefficients = firwin(M, ((passband_edge + stopband_edge) / (2*pi)), window=('kaiser', beta), pass_zero=True)
        frequencies, response = freqz(filter_coefficients, worN=10000)
        mag_res = 20 * log10(abs(response))
        print(f'Magnitude Response is:\n{mag_res}')

        plt.figure()
        plt.plot((frequencies), mag_res, label='Magnitude Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Magnitude Response (in dB)')
        plt.axvline(passband_edge, color='g', linestyle='dotted', label='Passband Edge (π/8)')
        plt.axvline(stopband_edge, color='r', linestyle='dotted', label='Stopband Edge (π/4)')
        plt.axhline(-40, color='g', linestyle='dotted', label='Passband Ripple (-40 dB)')
        plt.axhline(-50, color='r', linestyle='dotted', label='Stopband Ripple (-50 dB)')
        plt.legend()
        plt.grid(linestyle='dashed')
        plt.show()

        x=[]
        for i in range(0,16):
            x.append(2 * cos(pi / 16 * i) + 3 * cos(pi / 2 * i))
        n = [i for i in range(0, 16)]
        filtered_x = lfilter(filter_coefficients, 1.0, x)

        print(f'\nOriginal Signal x[n] is:\n{[f"{i:.4f}" for i in x]}')
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.stem(n, x, linefmt='b-', markerfmt='bo')
        for i, val in enumerate(x):
            plt.text(n[i], val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        plt.title('Original Signal x[n]')
        plt.xlabel('n')
        plt.ylabel('Amplitude')
        plt.grid(linestyle='-')

        print(f'\nFiltered Signal y[n] is:\n{[f"{i:.4f}" for i in filtered_x]}\n')
        plt.subplot(2, 1, 2)
        plt.stem(n, filtered_x, linefmt='g-', markerfmt='go')
        for i, val in enumerate(filtered_x):
            plt.text(n[i], val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        plt.title('Filtered Signal y[n]')
        plt.xlabel('n')
        plt.ylabel('Amplitude')
        plt.grid(linestyle='-')
        plt.tight_layout()
        plt.show()
        
    elif  question == 3:
        omega= 240
        pass_edge, stop_edge = array([40, 60, 100]), array([30, 80, 90])
        stop_ripple_db, pass_ripple_db = 25, 20
        stop_ripple, pass_ripple = 10**(-stop_ripple_db/20), 10**(-pass_ripple_db/20)

        max_ripple = max(stop_ripple,pass_ripple)
        stop_ripple, pass_ripple = max_ripple/stop_ripple, max_ripple/pass_ripple 
        weight = array([stop_ripple, pass_ripple, stop_ripple, pass_ripple])

        delta_f=min(abs(stop_edge[0]-pass_edge[0]), abs(stop_edge[1]-pass_edge[1]), abs(stop_edge[2]-pass_edge[2]))/omega
        M = (ceil((0.5*(stop_ripple_db+pass_ripple_db)-13)/(2.21*delta_f))+1)
        print(f'M is taken equal to: {int(M)}')
        M= 105  #105, 63 and 39

        bands, desired = multiply([0, stop_edge[0], pass_edge[0], pass_edge[1], stop_edge[1], stop_edge[2], pass_edge[2], omega/2], 1/omega), array([0, 1, 0, 1])
        filter_coefficient = remez(M, bands, desired, weight= weight)
        frequency, response = freqz(filter_coefficient, worN= 10000)
        frequency = frequency * (omega/(2*pi))
        magnitude_response = 20*log10(abs(response))
        print(f'Magnitude Response:\n{magnitude_response}\n')

        plt.figure()
        plt.plot(frequency, magnitude_response, label='Magnitude Response')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title('Magnitude Response (in dB)')
        plt.axvline(x=0, color='k', linestyle='dotted')
        plt.axvline(x=30, color='r', linestyle='dotted', label='Stop Band Edge')
        plt.axvline(x=40, color='g', linestyle='dotted', label='Pass Band Edge')
        plt.axvline(x=60, color='g', linestyle='dotted')
        plt.axvline(x=80, color='r', linestyle='dotted')
        plt.axvline(x=90, color='r', linestyle='dotted')
        plt.axvline(x=100, color='g', linestyle='dotted')
        plt.axvline(x=120, color='k', linestyle='dotted')
        plt.axhline(y=-20, color='purple', linestyle='dotted', label='Pass Band Ripple') 
        plt.axhline(y=-25, color='orange', linestyle='dotted', label='Stop Band Ripple')
        plt.hlines(20*log10(1+10**(-pass_ripple_db/20)), 0, omega/2, 'purple', 'dotted')
        plt.hlines(20*log10(1-10**(-pass_ripple_db/20)), 0, omega/2, 'purple', 'dotted')
        plt.legend()
        plt.grid(linestyle='dashed')
        plt.show()
        
    else:
        print("Thank You\n")