import { useQuery } from "@tanstack/react-query"
import { getCollections } from "@/lib/api/client"
import { DEMO_MODE } from "@/lib/demo"

export function useCollections() {
  return useQuery({
    queryKey: ["collections"],
    queryFn: getCollections,
    staleTime: 30_000,
    // No backend in a demo deployment -- see RightSidebar.tsx, which
    // shows an explanatory message instead of calling this at all.
    enabled: !DEMO_MODE,
  })
}
