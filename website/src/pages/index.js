import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

const findings = [
  {
    k: 'Γ ≈ 0,3',
    t: 'Безотрывный режим',
    d: 'Рабочая точка НИР лежит в безотрывном режиме; порог подбрасывания — лишь при f ≈ 30 Гц.',
  },
  {
    k: '5,5 дет/с',
    t: 'Граница устойчивости',
    d: 'Выше неё реактивная заслонка дестабилизирует поток; целевые 5 дет/с — у самой границы.',
  },
  {
    k: '×70',
    t: 'Дозирующий выпуск',
    d: 'Снижение дефектов с 96 до 1,4 на 1000 при потере производительности менее 4%.',
  },
  {
    k: '0,852',
    t: 'Выбор по TOPSIS',
    d: 'Дозирование — лидер многокритериального ранжирования, устойчивый к весам критериев.',
  },
];

const stages = [
  ['01', 'Вибротранспортирование', 'Численная модель механики детали на вибрирующем лотке'],
  ['02', 'Контур контроля', 'Дискретно-событийная модель потока и заслонки (SimPy)'],
  ['03', 'Стратегии управления', 'Три закона управления, сравнение по Парето'],
  ['04', 'Многокритериальный выбор', 'AHP с проверкой согласованности + TOPSIS'],
  ['05', 'What-if дашборд', 'Интерактивное исследование пространства параметров'],
  ['06', 'SA-документация', 'Концепция, требования, архитектура, отчёт'],
];

export default function Home() {
  return (
    <Layout title="FeederTwin" description="Имитационная модель системы поштучной подачи">
      <header className={styles.hero}>
        <div className={styles.heroInner}>
          <div className={styles.formula}>Γ = A·ω² / g</div>
          <h1 className={styles.title}>FeederTwin</h1>
          <p className={styles.tagline}>
            Имитационная модель и исследование стратегий управления
            автоматической системой поштучной подачи мелкоразмерных изделий.
            Развитие НИР НИТУ МИСИС: от чертежа — к модели и синтезу управления.
          </p>
          <div className={styles.cta}>
            <Link className={styles.btnPrimary} to="/sa/01-concept">SA-документация →</Link>
            <Link className={styles.btnGhost} to="/dashboard">What-if дашборд</Link>
          </div>
        </div>
      </header>

      <main>
        <section className={styles.section}>
          <h2 className={styles.h2}>Что модель сказала о системе</h2>
          <div className={styles.grid}>
            {findings.map((f) => (
              <div key={f.t} className={styles.card}>
                <div className={styles.cardK}>{f.k}</div>
                <div className={styles.cardT}>{f.t}</div>
                <div className={styles.cardD}>{f.d}</div>
              </div>
            ))}
          </div>
        </section>

        <section className={styles.sectionAlt}>
          <h2 className={styles.h2}>Шесть этапов</h2>
          <div className={styles.stages}>
            {stages.map(([n, t, d]) => (
              <div key={n} className={styles.stage}>
                <span className={styles.stageN}>{n}</span>
                <div>
                  <div className={styles.stageT}>{t}</div>
                  <div className={styles.stageD}>{d}</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </Layout>
  );
}
