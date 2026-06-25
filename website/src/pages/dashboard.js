import React from 'react';
import Layout from '@theme/Layout';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function Dashboard() {
  const src = useBaseUrl('/dashboard/index.html');
  return (
    <Layout title="What-if дашборд" description="Интерактивное исследование параметров FeederTwin">
      <iframe
        src={src}
        title="FeederTwin what-if дашборд"
        style={{ width: '100%', height: 'calc(100vh - 60px)', border: 0 }}
      />
    </Layout>
  );
}
