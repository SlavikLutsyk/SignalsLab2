import numpy as np
import matplotlib.pyplot as plt
import time

def compute_dft_term(f, k, N):
    """Обчислення k-го члена ряду Фур'є для ДПФ."""
    real_part = 0
    imag_part = 0
    for n in range(N):
        angle = 2 * np.pi * k * n / N
        real_part += f[n] * np.cos(angle)
        imag_part -= f[n] * np.sin(angle)
    return real_part + 1j * imag_part

def compute_dft(f, N):
    """Обчислення всіх коефіцієнтів DFT для сигналу f."""
    dft_result = [0]
    dft_result[0] = np.sum(f) / N
    for k in range(1,N):
        real_part = 0
        imag_part = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            real_part += f[n] * np.cos(angle)
            imag_part -= f[n] * np.sin(angle)
        real_part = real_part / N
        imag_part = imag_part / N
        dft_result.append(real_part + 1j * imag_part)
    return np.array(dft_result)

def compute_idft(dft_result, N):
    """Обчислення зворотного ДПФ (IDFT)."""
    reconstructed_signal = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            angle = 2 * np.pi * k * n / N
            reconstructed_signal[n] += dft_result[k] * np.exp(1j * angle)
    return np.real(reconstructed_signal)

def compute_dft_with_metrics(f, N):
    """Обчислення DFT разом із метриками часу та кількості операцій."""
    start_time = time.time()
    mul_count = 0
    add_count = 0
    dft_result = [0]

    dft_result[0] = np.sum(f) / N

    for k in range(1, N):
        real_part = 0
        imag_part = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            real_part += f[n] * np.cos(angle)
            imag_part -= f[n] * np.sin(angle)
            mul_count += 2  # Два множення (f[n] * cos та f[n] * sin)
            add_count += 1  # Одне додавання для кожної складової
        dft_result.append(real_part + 1j * imag_part)

    elapsed_time = time.time() - start_time
    return np.array(dft_result), elapsed_time, mul_count, add_count


def plot_spectrum(dft_result, N):
    """Побудова графіків амплітудного та фазового спектрів."""
    amplitudes = np.abs(dft_result)
    phases = np.angle(dft_result)

    # Графік амплітуд
    plt.figure(figsize=(10, 5))
    plt.stem(range(N), amplitudes, basefmt=" ")
    plt.title("Спектр амплітуд")
    plt.xlabel("k")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.show()

    # Графік фаз
    plt.figure(figsize=(10, 5))
    plt.stem(range(N), phases, basefmt=" ")
    plt.title("Фазовий спектр")
    plt.xlabel("k")
    plt.ylabel("Фаза (радіани)")
    plt.grid(True)
    plt.show()

def reconstruct_signal(dft_result, N, t_values):
    """Відтворення аналогового сигналу за допомогою зворотного ДПФ."""
    reconstructed_signal = np.zeros_like(t_values, dtype=np.float64)

    for k in range(N):
        C_k = dft_result[k]
        for n, t in enumerate(t_values):
            reconstructed_signal[n] += np.real(C_k * np.exp(1j * 2 * np.pi * k * t / N))

    return reconstructed_signal

def get_signal_expression(dft_result, N):
    expression = ""
    for k, C_k in enumerate(dft_result):
        amplitude = np.abs(C_k)
        phase = np.angle(C_k)
        expression += f" + {amplitude:.5f} * cos(2π * {k} * t / T + {phase:.2f})"
    return expression.strip(" +")

def s_table(dft_res, t_T):
    s = 0
    for k, C_k in enumerate(dft_res):
        amplitude = np.abs(C_k)
        phase = np.angle(C_k)
        s += amplitude * np.cos(2 * np.pi * k * t_T + phase)
    return s

def plot_time_domain_signal(t_values, reconstructed_signal):
    """Побудова графіку відтвореного сигналу у часовій області."""
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, reconstructed_signal, label="Відтворений сигнал")
    plt.title("Часова залежність відтвореного сигналу")
    plt.xlabel("Час (t)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.legend()
    plt.show()


# Частина 1: DFT для довільного сигналу
def part_one():
    print("\n\nЧастина 1\n")
    N = 10 + 5

    f = np.random.rand(N)
    print(f"Довільний сигнал: {f}")

    dft_result, elapsed_time, mul_count, add_count = compute_dft_with_metrics(f, N)

    print(f"\nЧас обчислення ДПФ: {elapsed_time:.6f} секунд")
    print(f"\nКількість операцій множення: {mul_count}")
    print(f"\nКількість операцій додавання: {add_count}")

    print("\nКоефіцієнти DFT (C_k):")
    for k in range(N):
        print(f"C_{k}: {dft_result[k]:.3f}")

    plot_spectrum(dft_result, N)


# Частина 2: Відтворення аналогового сигналу
def part_two():
    N = 96 + 5
    binary_representation = bin(N)[2:].zfill(7)
    f = [1] + [int(bit) for bit in binary_representation]

    print("\n\nЧастина 2: Відтворення аналогового сигналу\n")
    print(f"Сигнал, створений з двійкових чисел для N={N}: {f}")

    # Обчислення DFT
    dft_result = compute_dft(f, len(f))

    # Отримання амплітуд та фаз
    amplitudes = np.abs(dft_result)
    phases = np.angle(dft_result)

    print("\nКоефіцієнти DFT (C_n):")
    for k, C_k in enumerate(dft_result):
        print(f"C_{k}: {C_k:.3f}")

    print("\nАмплітуди |C_n|:")
    for k, amplitude in enumerate(amplitudes):
        print(f"|C_{k}|: {amplitude:.5f}")

    print("\nФази arg(C_n):")
    for k, phase in enumerate(phases):
        print(f"arg(C_{k}): {phase:.5f} ")

    # Відтворення сигналу
    t_values = np.linspace(0, len(f) - 1, 1000)
    reconstructed_signal = reconstruct_signal(dft_result, len(f), t_values)
    signal_expression = get_signal_expression(dft_result, len(f))
    print(f"\nВідтворений сигнал можна виразити як:\ns(t) = {signal_expression}")

    for t in np.arange(0, 1 + 1/8, 1/8):
       print(f"{t} - {s_table(dft_result, t)}")

    # Побудова графіка відтвореного сигналу
    plot_time_domain_signal(t_values, reconstructed_signal)

    return f

# Частина 3: Обчислення значень відліків s(nTδ) для n = 0, 1.
def part_three(f):
    """Частина 3: Обчислення значень відліків s(nTδ) для n = 0, 1."""
    N = 8

    # Обчислення DFT
    dft_result = compute_dft(f, N)

    # Обернене ДПФ
    idft_result = compute_idft(dft_result, N)

    # Вивід значень s(nTδ) для n = 0, 1.
    s_n = []
    print(f"\nЧастина 3\n")
    for i in range(8):
        s_n.append(idft_result[i])
        print(f"Значення s({i}Tδ): {abs(round(s_n[i], 0))}")

    # Аналітичні вирази для s(nTδ) при n = 0, 1
    for n in range(2):
        terms = []
        for k in range(N):
            amplitude = abs(dft_result[k])
            phase = np.angle(dft_result[k])
            term = f"{amplitude:.2f} * e^(j * ({phase:.2f} + (2π * {k} * {n} / {N})))"
            terms.append(term)
        expression = " + ".join(terms)
        print(f"\nАналітичний вираз s({n}Tδ): {expression}")


def main():
    part_one()
    f = part_two()
    part_three(f)

if __name__ == "__main__":
    main()