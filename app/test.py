"""
resolver_equacao_11.py

Script "muito complexo" (intencionalmente verboso) que explica, passo a passo e com
várias abordagens, como resolver a equação linear simples:

    11 = 11 * 1/5 + b

O objetivo é demonstrar (com aritmética de frações exata, decimais de alta precisão,
verificação, e — se disponível — solução simbólica via sympy) como isolar e calcular
b, e também apresentar explicações textuais detalhadas em português.

Como usar:
    python resolver_equacao_11.py

Dependências opcionais:
    - sympy (apenas se quiser a seção simbólica automática)

Este código é propositalmente longuíssimo para ensinar cada pequeno detalhe.
"""

from fractions import Fraction
from decimal import Decimal, getcontext
import math
import os
import sys

# Tentativa de importar sympy para demonstração simbólica (opcional)
try:
    import sympy as sp
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False


def pretty_fraction(fr: Fraction) -> str:
    """Retorna uma string legível para a fração, reduzida e também como número misto."""
    num, den = fr.numerator, fr.denominator
    if abs(num) >= den:
        whole = num // den
        rem = abs(num % den)
        if rem == 0:
            return f"{whole}"
        else:
            return f"{whole} {rem}/{den}"
    else:
        return f"{num}/{den}"


def step_by_step_fractional_solution():
    """Resolve usando aritmética de frações (exata) e imprime cada passo."""
    print("=== Solução passo a passo (frações exatas) ===\n")

    # 1) Identificar os termos
    lhs = Fraction(11, 1)  # 11
    multiplicador = Fraction(11, 1)  # 11 (coeficiente)
    fator = Fraction(1, 5)  # 1/5

    print(f"Equação original: 11 = 11 * 1/5 + b")
    print(f"Lado esquerdo (LHS): {lhs}")
    print(f"Termo multiplicativo: {multiplicador} * {fator}")

    # 2) Calcular o produto 11 * 1/5 como fração exata
    produto = multiplicador * fator
    print(f"Passo: calcular o produto do coeficiente pelo fator:\n  11 * 1/5 = {produto} (como fração) -> {pretty_fraction(produto)})")

    # 3) Isolar b: subtrair o produto de ambos os lados
    b = lhs - produto
    print("\nIsolando b: subtrair '11 * 1/5' dos dois lados:")
    print(f"  b = LHS - produto = {lhs} - {produto} = {b}")

    # 4) Mostrar a forma reduzida e mista
    mixed = pretty_fraction(b)
    print(f"Forma exata (reduzida): {b} -> {mixed}")

    # 5) Forma decimal (com precisão controlada)
    getcontext().prec = 28  # 28 casas decimais de precisão
    dec_lhs = Decimal(11)
    dec_prod = Decimal(11) * (Decimal(1) / Decimal(5))
    dec_b = dec_lhs - dec_prod
    print(f"Forma decimal com Decimal(prec=28): {dec_b} (exata dentro da precisão definida)")
    print(f"Forma decimal curta (float): {float(dec_b)}")

    print("\nVerificação: somar o produto com b e comparar com LHS:")
    check = produto + b
    print(f"  produto + b = {produto} + {b} = {check}")
    print(f"  LHS = {lhs}")
    print(f"  Igualdade verificada? -> {check == lhs}")

    return b


def algebraic_explanation_textual():
    """Explica, em linguagem natural, os passos algébricos usados para isolar b."""
    print("\n=== Explicação algébrica em palavras (português) ===\n")
    print("1) Começamos com a equação: 11 = 11 * 1/5 + b")
    print("2) Identificamos que o termo que contém b está isolado no lado direito: 'b'.")
    print("3) Para isolar b, subtraímos do lado direito o termo '11 * 1/5' e fazemos o mesmo no lado esquerdo:\n   b = 11 - (11 * 1/5)")
    print("4) Calculamos 11 * 1/5 = 11/5. Portanto b = 11 - 11/5.")
    print("5) Convertendo 11 para quintos: 11 = 55/5, então b = 55/5 - 11/5 = 44/5.")
    print("6) Conclusão: b = 44/5 = 8.8")


def symbolic_solution_if_available():
    """Tenta usar sympy para resolver simbolicamente; se sympy não existir, informa ao usuário."""
    print("\n=== Solução simbólica (via sympy) ===\n")
    if not _HAS_SYMPY:
        print("Sympy não está disponível neste ambiente. Pule esta seção ou instale sympy para ver a saída simbólica.")
        print("Para instalar: pip install sympy")
        return None

    # Usando sympy para montar e resolver a equação
    b = sp.symbols('b')
    eq = sp.Eq(11, 11 * sp.Rational(1, 5) + b)
    print(f"Equação simbólica montada: {eq}")

    sol = sp.solve(eq, b)
    print(f"Solução simbólica (lista): {sol}")
    if sol:
        print(f"Forma simbólica simplificada: b = {sp.nsimplify(sol[0])}")
    return sol


def alternative_checks(b_value):
    """Faz checagens alternativas (aproximação, representação mista, checagem por float)."""
    print("\n=== Checagens alternativas ===\n")

    # Representações
    fract = b_value if isinstance(b_value, Fraction) else Fraction(b_value)
    print(f"Representação exata (fração): {fract} -> {pretty_fraction(fract)}")

    # Mixed number
    whole = fract.numerator // fract.denominator
    rem = abs(fract.numerator % fract.denominator)
    print(f"Número misto: {whole} e restante {rem}/{fract.denominator} (se aplicável)")

    # Float check
    float_val = float(fract)
    print(f"Representação float: {float_val}")

    # Tolerância para comparação (float)
    lhs = 11.0
    rhs = float(11.0 * (1.0 / 5.0) + float_val)
    print(f"Checagem com floats: LHS={lhs}, RHS (recalculado)={rhs}, diferença={lhs - rhs}")
    print(f"Dentro de tolerância de ponto flutuante? -> {math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)}")


def run_all_and_save_log(save_path=None):
    """Executa todas as partes e (opcional) salva uma saída de log em arquivo de texto."""
    lines_before = []

    # Redirecionar saída para string simples (simplesmente construímos uma lista de linhas)
    # Em vez de redirect de stdout para capturar prints (para manter simples), chamamos as funções
    # e permitimos que imprimam no console. Se save_path for fornecido, também salvaremos uma cópia.

    print("\n>> Iniciando execução completa - explicações detalhadas:\n")
    b_frac = step_by_step_fractional_solution()
    algebraic_explanation_textual()
    if _HAS_SYMPY:
        symbolic_solution_if_available()
    else:
        print('\n(secção simbólica omitida: sympy ausente)')

    alternative_checks(b_frac)

    # Salvar saída reduzida em arquivo (apenas resumo)
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('Resumo da solução:\n')
            f.write(f'b (exata) = {b_frac}\n')
            f.write(f'b (misto) = {pretty_fraction(b_frac)}\n')
            f.write(f'b (decimal) = {float(b_frac)}\n')
        print(f"\nResumo salvo em: {save_path}")

    return b_frac


if __name__ == '__main__':
    # Local padrão para salvar um pequeno resumo na Área de Trabalho
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    if not os.path.isdir(desktop):
        # fallback para diretório atual
        desktop = os.getcwd()
    resumo_path = os.path.join(desktop, 'resumo_solucao_11.txt')

    print('Script: "resolver_equacao_11.py" — explicando detalhadamente como resolver 11 = 11*1/5 + b')
    resultado = run_all_and_save_log(save_path=resumo_path)

    print('\n--- Conclusão final (resumida) ---')
    print(f"b = {resultado}  (fração exata)")
    print(f"b = {pretty_fraction(resultado)}  (forma mista, se aplicável)")
    print(f"b = {float(resultado)}  (decimal)")
    print('\nO resumo curta foi salvo em: ' + resumo_path)
