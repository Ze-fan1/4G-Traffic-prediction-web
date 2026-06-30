import { useState, useCallback } from 'react';
import Header from './components/Header';
import Loader from './components/Loader';
import OverviewPage from './pages/OverviewPage';
import PlaygroundPage from './pages/PlaygroundPage';
import ComparePage from './pages/ComparePage';
import ErrorsPage from './pages/ErrorsPage';
import DetailsPage from './pages/DetailsPage';

const PAGES = {
  overview: OverviewPage,
  playground: PlaygroundPage,
  compare: ComparePage,
  errors: ErrorsPage,
  details: DetailsPage,
};

export default function App() {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  const handleLoadDone = useCallback(() => setLoading(false), []);
  const PageComponent = PAGES[activeTab];

  return (
    <>
      {loading && <Loader onDone={handleLoadDone} />}
      <Header activeTab={activeTab} onTabChange={setActiveTab} />
      <main
        className="max-w-7xl mx-auto px-3 sm:px-5 pb-24"
        style={{ opacity: loading ? 0 : 1, transition: 'opacity 0.5s' }}
      >
        <PageComponent key={activeTab} />
      </main>
    </>
  );
}
