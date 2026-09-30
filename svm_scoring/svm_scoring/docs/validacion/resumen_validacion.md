# Validación de la base de datos simulada

Registros simulados: **6,500** (mínimo exigido: 5.000)

## 1. Balance de la variable objetivo

- Simulado: bueno=77.5%, malo=22.5% (criterio de la Primera entrega: 20%-25% de 'malo').
- German Credit original: bueno=70.0%, malo=30.0%.

## 2. Relaciones variable-objetivo (monotonicidad)

Criterio: la tasa de 'malo' debe crecer estrictamente en el orden de riesgo esperado.

| Variable | Orden esperado (menor -> mayor riesgo) | Tasa de 'malo' | ¿Cumple? |
|---|---|---|---|
| saldo_categoria | alto < medio < bajo < sin_cuenta | 9.8%, 14.2%, 23.2%, 30.1% | Sí |
| estado_pago_previo | al_dia < sin_creditos_previos < mora_leve < mora_severa | 13.0%, 24.2%, 27.5%, 37.9% | Sí |
| nivel_ahorro | muy_alto < alto < medio < bajo | 14.9%, 15.5%, 20.8%, 24.0% | Sí |
| tipo_ocupacion | alta_calificacion < formal_calificado < formal_no_calificado < informal | 12.0%, 20.1%, 20.6%, 26.9% | Sí |
| Cuartil de monto | Q1 < Q2 < Q3 < Q4 | 13.7%, 18.5%, 24.4%, 33.2% | Sí |
| Cuartil de plazo | Q1 < Q2 < Q3 < Q4 | 14.4%, 18.7%, 25.0%, 35.3% | Sí |

## 3. Forma de las distribuciones numéricas (simulado vs. original)

Criterio: asimetría positiva en las tres variables y coeficiente de variación (CV) a menos de 0.10 del original.

| Variable | Media sim. | CV sim. | Asimetría sim. | Media orig. | CV orig. | Asimetría orig. |
|---|---|---|---|---|---|---|
| Edad (años) | 35.5 | 0.31 | 1.11 | 35.5 | 0.32 | 1.02 |
| Plazo (meses) | 20.9 | 0.55 | 1.28 | 20.9 | 0.58 | 1.09 |
| Monto (COP vs. DM) | 5,118,505.2 | 0.83 | 2.83 | 3,271.3 | 0.86 | 1.95 |

## 4. Estructura de dependencia

- Correlación de Pearson plazo-monto: simulado **0.603**, original (duration-credit_amount) **0.625**. Criterio: diferencia < 0.05.

## 5. Proporciones de categorías calibradas contra el original

| Variable simulada | Categoría | Simulado | Original (mapeado) |
|---|---|---|---|
| saldo_categoria | sin_cuenta | 38.7% | 39.4% |
| saldo_categoria | bajo | 27.1% | 27.4% |
| saldo_categoria | medio | 27.2% | 26.9% |
| saldo_categoria | alto | 7.0% | 6.3% |
| estado_pago_previo | al_dia | 53.4% | 53.0% |
| estado_pago_previo | mora_severa | 28.7% | 29.3% |
| estado_pago_previo | sin_creditos_previos | 8.9% | 8.9% |
| estado_pago_previo | mora_leve | 9.1% | 8.8% |

`nivel_ahorro` se calibró con A61-A64 redistribuyendo A65 (desconocido); `tipo_vivienda` y `tipo_ocupacion` son supuestos del contexto colombiano.

## 6. Versión con faltantes

| Columna | % faltante |
|---|---|
| edad | 3.32% |
| tipo_ocupacion | 4.32% |
| nivel_ahorro | 5.00% |
| saldo_cuenta_cop | 4.78% |
| monto_credito_cop | 3.77% |
| plazo_meses | 2.94% |

Saldos faltantes en clientes `sin_cuenta` (debe ser 0): 0

## Resumen de verificaciones

| Verificación | Resultado |
|---|---|
| Tamaño >= 5.000 | CUMPLE |
| Balance 20%-25% de 'malo' | CUMPLE |
| Monotonicidad saldo_categoria | CUMPLE |
| Monotonicidad estado_pago_previo | CUMPLE |
| Monotonicidad nivel_ahorro | CUMPLE |
| Monotonicidad tipo_ocupacion | CUMPLE |
| Monotonicidad monto_credito_cop | CUMPLE |
| Monotonicidad plazo_meses | CUMPLE |
| Forma edad | CUMPLE |
| Forma plazo_meses | CUMPLE |
| Forma monto_credito_cop | CUMPLE |
| Correlación plazo-monto a ±0.05 del original | CUMPLE |
| Faltantes entre 2.5% y 5.5% por columna | CUMPLE |
| Sin saldos faltantes en 'sin_cuenta' | CUMPLE |
