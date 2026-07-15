import { useQuery } from "@tanstack/react-query"
import { getProviders } from "@/lib/api/client"
import { DEMO_MODE } from "@/lib/demo"

export function useProviders() {
  return useQuery({
    queryKey: ["providers"],
    queryFn: getProviders,
    staleTime: 60_000,
    // No backend in a demo deployment -- ResponseModeSection forces Web
    // search / Think / Agent off in demo mode regardless of this.
    enabled: !DEMO_MODE,
  })
}
