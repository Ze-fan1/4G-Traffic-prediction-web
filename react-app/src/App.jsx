import { useState, useCallback, useEffect } from 'react';
import Header from './components/Header';
import Loader from './components/Loader';
import OverviewPage from './pages/OverviewPage';
import PerformancePage from './pages/PerformancePage';
import CurvesPage from './pages/CurvesPage';
import ErrorsPage from './pages/ErrorsPage';
import DetailsPage from './pages/DetailsPage';

const PAGES = {
  overview: OverviewPage,
  performance: PerformancePage,
  curves: CurvesPage,
  errors: ErrorsPage,
  details: DetailsPage,
};

export default function App() {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  // Safety: force show content after 3s even if loader doesn't fire
  useEffect(() => {
    const t = setTimeout(() => setLoading(false), 3000);
    return () => clearTimeout(t);
  }, []);

  const handleLoadDone = useCallback(() => setLoading(false), []);
  const PageComponent = PAGES[activeTab];

  return (
    <>
      {loading && <Loader onDone={handleLoadDone} />}
      <Header activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="max-w-7xl mx-auto px-3 sm:px-5 pb-24">
        <PageComponent key={activeTab} />
      </main>
    </>
  );
}
