from numpy import array, pi, tan, sqrt, log, log10, ceil, zeros
from scipy.signal import cheby1, lp2bs, bilinear, freqz, butter, lp2hp
from matplotlib.pyplot import figure, plot, axvline, axhline, title, xlabel, ylabel, show, legend, grid, ylim

question = 1
while question in [1, 2]:
    question = int(input("Please enter a Question 1 or 2: "))
    print()
    
    if  question == 1: 
        passband_edge , stopband_edge = 1/3 , 1/6
        pass_ripple_db , stop_ripple_db = 10, 60
        omega_pass , omega_stop = 1 / tan(pi * passband_edge / 2) , 1 / tan(pi * stopband_edge / 2)

        filter_coefficient  = int(ceil(log10((10**(stop_ripple_db / 10) - 1) / (10**(pass_ripple_db / 10) - 1)) / (2 * log10(omega_stop / omega_pass))))
        Wn = omega_pass / ((10**(pass_ripple_db / 10) - 1)**(1 / (2 * filter_coefficient)))

        b_analog_lp, a_analog_lp = butter(filter_coefficient, omega_pass, analog=True, btype='low')
        b_analog_hp, a_analog_hp = lp2hp(b_analog_lp, a_analog_lp, wo=omega_pass)
        b_digital_hp, a_digital_hp = bilinear(b_analog_hp, a_analog_hp)

        frequecy, h = freqz(b_digital_hp, a_digital_hp, worN=8000)
        mag_res= 20 * log10(abs(h))

        figure()
        plot(frequecy, mag_res, label='Magnitude Response')
        axvline(passband_edge*pi, color='g', linestyle='--', label='Passband Edge')
        axvline(stopband_edge*pi, color='r', linestyle='--', label='Stopband Edge')
        axhline(-stop_ripple_db, color='m', linestyle='--', label='Stopband Ripple')
        axhline(20 * log10(1+10**(-pass_ripple_db / 20)), color='y', linestyle='--', label='1 + Delta')
        axhline(20 * log10(1-10**(-pass_ripple_db / 20)), color='y', linestyle='--', label='1 - Delta')
        ylim(-200, 20)
        title('IIR High-Pass Filter')
        xlabel('Angular Frequency (radians/sample)')
        ylabel('Magnitude (dB)')
        grid()
        legend()
        show()
        
    elif  question == 2:
        
        pass_ripple_db, stop_ripple_db = 30, 40
        passband_edges, stopband_edges = array([pi/6, 2*pi/3]), array([pi / 4, pi / 2])
        omega_pass, omega_stop = 2 * tan(passband_edges / 2), 2 * tan(stopband_edges / 2)
        B, Omega_0= omega_pass[1]-omega_pass[0], sqrt(omega_pass[0]*omega_pass[1])
        pass_ripple, stop_ripple = 10**(-pass_ripple_db/20), 10**(-stop_ripple_db/20)

        passband_attenuation = -20*log10(1 - pass_ripple)
        Omega_pass_lpf, Omega_stop_lpf = 1, min(omega_stop[1] * B / (omega_stop[1]**2 - Omega_0**2), omega_stop[0] * B / (Omega_0**2 - omega_stop[0]**2))
        N = ceil(log((sqrt(1-stop_ripple**2)+ sqrt(1-stop_ripple**2*(1+(sqrt(1/(1-pass_ripple)**2-1))**2)))/(stop_ripple*(sqrt(1/(1-pass_ripple)**2-1))))/ log(Omega_stop_lpf / Omega_pass_lpf + sqrt((Omega_stop_lpf / Omega_pass_lpf)**2 - 1)))

        b_analog_lp, a_analog_lp = cheby1(N, passband_attenuation, 1, analog=True)
        b_analog_bsp, a_analog_bsp = lp2bs(b_analog_lp, a_analog_lp, Omega_0, B)
        b_digital_bsp, a_digital_bsp = bilinear(b_analog_bsp, a_analog_bsp)
        frequency, h = freqz(b_digital_bsp, a_digital_bsp, worN=8000)
        mag_response = 20 * log10(abs(h))

        figure()
        plot(frequency, mag_response, label='Magnitude Response')
        axhline(0, color='black', linestyle='-', label='0 dB')
        axvline(passband_edges[0], color='g', linestyle='--', label='Passband Edges')
        axvline(stopband_edges[0], color='r', linestyle='--', label='Stopband Edges')
        axhline(-stop_ripple_db, color='y', linestyle='--', label='Stopband Ripple')
        axhline(-pass_ripple_db, color='b', linestyle='--', label='Passband Ripple')
        axvline(passband_edges[1], color='g', linestyle='--')
        axvline(stopband_edges[1], color='r', linestyle='--')
        title('Magnitude Response of IIR Knotch Filter')
        xlabel('Angular Frequency (radians/sample)')
        ylabel('Magnitude (dB)')
        legend(loc='best')
        grid(linestyle='dashed')
        show()

        lenght = 3
        b_comb = zeros(lenght * len(b_digital_bsp) - (lenght-1))
        a_comb = zeros(lenght * len(a_digital_bsp) - (lenght-1))
        b_comb[::lenght], a_comb[::lenght] = b_digital_bsp, a_digital_bsp
        frequency_comb, h_comb = freqz(b_comb, a_comb, worN=8000)
        mag_response_comb = 20 * log10(abs(h_comb))

        figure()
        plot(frequency_comb, mag_response_comb, label='Magnitude Response(L=3)')
        xlabel('Angular Frequency (radians/sample)')
        ylabel(r'Magnitude (dB)')
        title(f'Magnitude Response of IIR Knotch Filter (L = {lenght})')
        legend(loc='best')
        grid(linestyle='dashed')
        show()
        
    else:
        print("Thank You\n")